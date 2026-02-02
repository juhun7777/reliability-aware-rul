"""
Condition-Specific Ensemble LSTM Model
IEEE Paper: Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins

Paper Section III-D: Condition-Specific Bi-LSTM Ensemble
- 3 independent models for 0A, 0.5A, 1.0A conditions
- STFT-based soft weighting (Paper Algorithm 3)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.base_model import BaseDeepLearningModel
from models.lstm import LSTMNetwork
from utils.metrics import calculate_rmse
from utils.preprocessing import inverse_transform_rul


class EnsembleLSTMModel(BaseDeepLearningModel):
    """
    Condition-Specific Ensemble LSTM (Paper Section III-D, Algorithm 3)

    - 3 independent LSTM models (0A, 0.5A, 1.0A)
    - Each model includes Bahdanau attention
    - Soft label based weighted averaging (Equation 8)
    """

    def __init__(self, config):
        super().__init__(config)
        self.build_model()

    def build_model(self):
        """Create 3 condition-specific LSTM models"""
        # ModuleDict keys cannot contain '.' so use '_'
        self.models = nn.ModuleDict({
            '0A': LSTMNetwork(
                input_size=5,
                hidden_size=self.config.LSTM_HIDDEN_SIZE,
                num_layers=self.config.LSTM_NUM_LAYERS,
                dropout=self.config.DROPOUT,
                bidirectional=self.config.LSTM_BIDIRECTIONAL
            ).to(self.device),
            '05A': LSTMNetwork(
                input_size=5,
                hidden_size=self.config.LSTM_HIDDEN_SIZE,
                num_layers=self.config.LSTM_NUM_LAYERS,
                dropout=self.config.DROPOUT,
                bidirectional=self.config.LSTM_BIDIRECTIONAL
            ).to(self.device),
            '10A': LSTMNetwork(
                input_size=5,
                hidden_size=self.config.LSTM_HIDDEN_SIZE,
                num_layers=self.config.LSTM_NUM_LAYERS,
                dropout=self.config.DROPOUT,
                bidirectional=self.config.LSTM_BIDIRECTIONAL
            ).to(self.device)
        })

        # Condition key mapping
        self.condition_keys = {'0A': '0A', '0.5A': '05A', '1.0A': '10A'}
        self.condition_list = ['0A', '0.5A', '1.0A']

        # Huber Loss for each model (Paper Equation 11: delta = 0.5)
        self.criterions = {
            key: nn.HuberLoss(delta=0.5)
            for key in ['0A', '05A', '10A']
        }

        # AdamW Optimizer (Paper Table IV)
        self.optimizers = {
            key: optim.AdamW(
                self.models[key].parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=1e-4
            )
            for key in ['0A', '05A', '10A']
        }

        # Learning Rate Scheduler
        self.schedulers = {
            key: optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[key],
                mode='min',
                factor=0.5,
                patience=self.config.PATIENCE // 2
            )
            for key in ['0A', '05A', '10A']
        }

    def _get_model_key(self, condition):
        """Convert condition name to model key"""
        return self.condition_keys.get(condition, condition)

    def fit(self, train_loader, val_loader=None):
        """
        Ensemble Training - Train each condition-specific model independently
        Paper Algorithm 3: Condition-Specific Bi-LSTM Ensemble Prediction
        """
        print(f"\n[Ensemble LSTM] Training started...")
        print("="*70)

        best_val_rmse = {key: float('inf') for key in ['0A', '05A', '10A']}
        best_epoch = {key: 0 for key in ['0A', '05A', '10A']}

        history = {
            'train_loss': {key: [] for key in ['0A', '05A', '10A']},
            'val_loss': {key: [] for key in ['0A', '05A', '10A']},
            'val_rmse_orig': {key: [] for key in ['0A', '05A', '10A']}
        }

        for epoch in range(self.config.EPOCHS):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            print(f"{'='*70}")

            # Train each condition-specific model
            for condition in self.condition_list:
                model_key = self._get_model_key(condition)
                self.models[model_key].train()

                train_losses = []
                train_preds = []
                train_targets = []

                for batch in train_loader:
                    inputs = batch['inputs'].to(self.device)
                    targets = batch['rul'].to(self.device)
                    soft_labels = batch['soft_label'].to(self.device)

                    # Get weight for current condition from soft labels
                    condition_idx = self.condition_list.index(condition)
                    condition_weight = soft_labels[:, condition_idx]

                    # Forward pass
                    outputs = self.models[model_key](inputs)

                    # Weighted loss (soft label weighting)
                    loss = self.criterions[model_key](outputs.squeeze(), targets)
                    weighted_loss = (loss * condition_weight).mean()

                    # Backward pass
                    self.optimizers[model_key].zero_grad()
                    weighted_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.models[model_key].parameters(), max_norm=1.0)
                    self.optimizers[model_key].step()

                    train_losses.append(weighted_loss.item())
                    train_preds.append(outputs.detach().cpu().numpy())
                    train_targets.append(targets.cpu().numpy())

                train_loss = np.mean(train_losses)
                train_preds = np.concatenate(train_preds).squeeze()
                train_targets = np.concatenate(train_targets)

                # Validation
                if val_loader is not None:
                    val_loss, val_preds, val_targets = self._validate_single_model(
                        condition, val_loader
                    )

                    # Original space RMSE
                    val_preds_orig = inverse_transform_rul(
                        val_preds, self.config.MAX_RUL_CYCLE, self.config.MIN_RUL_CYCLE
                    )
                    val_targets_orig = inverse_transform_rul(
                        val_targets, self.config.MAX_RUL_CYCLE, self.config.MIN_RUL_CYCLE
                    )
                    val_rmse_orig = calculate_rmse(val_preds_orig, val_targets_orig)

                    # Scheduler update
                    self.schedulers[model_key].step(val_rmse_orig)

                    # Best model check
                    if val_rmse_orig < best_val_rmse[model_key]:
                        best_val_rmse[model_key] = val_rmse_orig
                        best_epoch[model_key] = epoch
                        model_path = self.config.MODEL_PATH.replace('.pth', f'_{condition}.pth')
                        torch.save(self.models[model_key].state_dict(), model_path)
                        best_marker = "v"
                    else:
                        best_marker = " "

                    # Record history
                    history['train_loss'][model_key].append(train_loss)
                    history['val_loss'][model_key].append(val_loss)
                    history['val_rmse_orig'][model_key].append(val_rmse_orig)

                    print(f"{best_marker} {condition:>5} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
                          f"Val RMSE: {val_rmse_orig:.1f} cycles")

        print("\n" + "="*70)
        print("Ensemble training completed!")
        for condition in self.condition_list:
            model_key = self._get_model_key(condition)
            print(f"  {condition}: Best Epoch {best_epoch[model_key]+1}, "
                  f"Best Val RMSE: {best_val_rmse[model_key]:.1f} cycles")
        print("="*70)

        return history

    def _validate_single_model(self, condition, val_loader):
        """Validate single condition model"""
        model_key = self._get_model_key(condition)
        self.models[model_key].eval()

        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(self.device)
                targets = batch['rul'].to(self.device)
                soft_labels = batch['soft_label'].to(self.device)

                condition_idx = self.condition_list.index(condition)
                condition_weight = soft_labels[:, condition_idx]

                outputs = self.models[model_key](inputs)
                loss = self.criterions[model_key](outputs.squeeze(), targets)
                weighted_loss = (loss * condition_weight).mean()

                val_losses.append(weighted_loss.item())
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        return np.mean(val_losses), np.concatenate(all_preds).squeeze(), np.concatenate(all_targets)

    def predict(self, dataloader):
        """
        Ensemble Prediction - Soft label weighted average (Paper Equation 8)
        y_final = sum(w_c * y_c)
        """
        for key in ['0A', '05A', '10A']:
            self.models[key].eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['inputs'].to(self.device)
                targets = batch['rul'].cpu().numpy()
                soft_labels = batch['soft_label'].cpu().numpy()

                # Predictions from each condition model
                predictions_per_condition = []
                for condition in self.condition_list:
                    model_key = self._get_model_key(condition)
                    pred = self.models[model_key](inputs).cpu().numpy()
                    predictions_per_condition.append(pred)

                # (batch, 3)
                predictions_per_condition = np.concatenate(predictions_per_condition, axis=1)

                # Weighted average (Equation 8)
                ensemble_pred = np.sum(soft_labels * predictions_per_condition, axis=1)

                all_predictions.append(ensemble_pred)
                all_targets.append(targets)

        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        return predictions, targets

    def predict_with_uncertainty(self, dataloader, n_iter=30):
        """
        Monte Carlo Dropout for Uncertainty Quantification
        Paper Section III-E, Equation 9-10
        """
        for key in ['0A', '05A', '10A']:
            self.models[key].eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['inputs'].to(self.device)
                targets = batch['rul'].cpu().numpy()
                soft_labels = batch['soft_label'].cpu().numpy()

                # MC Dropout iterations
                mc_predictions = []
                for _ in range(n_iter):
                    predictions_per_condition = []
                    for condition in self.condition_list:
                        model_key = self._get_model_key(condition)
                        pred = self.models[model_key](inputs, mc_dropout=True).cpu().numpy()
                        predictions_per_condition.append(pred)

                    predictions_per_condition = np.concatenate(predictions_per_condition, axis=1)
                    ensemble_pred = np.sum(soft_labels * predictions_per_condition, axis=1)
                    mc_predictions.append(ensemble_pred)

                mc_predictions = np.array(mc_predictions)

                all_predictions.append(mc_predictions)
                all_targets.append(targets)

        all_predictions = np.concatenate(all_predictions, axis=1)
        all_targets = np.concatenate(all_targets)

        predictions_mean = np.mean(all_predictions, axis=0)
        predictions_std = np.std(all_predictions, axis=0)

        return predictions_mean, predictions_std, all_predictions, all_targets

    def save(self, path):
        """Save ensemble models"""
        for condition in self.condition_list:
            model_key = self._get_model_key(condition)
            model_path = path.replace('.pth', f'_{condition}.pth')
            torch.save(self.models[model_key].state_dict(), model_path)
        print(f"  Ensemble models saved")

    def load(self, path):
        """Load ensemble models"""
        for condition in self.condition_list:
            model_key = self._get_model_key(condition)
            model_path = path.replace('.pth', f'_{condition}.pth')
            self.models[model_key].load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"  Ensemble models loaded")

    def get_model_name(self):
        return "Ensemble Bi-LSTM (Condition-Specific)"

    def extract_attention_weights(self, dataloader):
        """Extract attention weights for each condition model"""
        attention_dict = {cond: [] for cond in self.condition_list}

        for key in ['0A', '05A', '10A']:
            self.models[key].eval()

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['inputs'].to(self.device)

                for condition in self.condition_list:
                    model_key = self._get_model_key(condition)
                    _ = self.models[model_key](inputs)

                    if hasattr(self.models[model_key].attention, 'weights'):
                        weights = self.models[model_key].attention.weights.cpu().numpy()
                        attention_dict[condition].append(weights.squeeze())

        for condition in self.condition_list:
            if attention_dict[condition]:
                attention_dict[condition] = np.concatenate(attention_dict[condition], axis=0)

        return attention_dict
