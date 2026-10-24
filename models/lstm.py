"""
LSTM Model for RUL Prediction
IEEE Paper: Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins

Implements Bahdanau-style Attention Mechanism (Paper Equation 6-7)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.base_model import BaseDeepLearningModel
from utils.metrics import calculate_rmse
from utils.preprocessing import inverse_transform_rul


class BahdanauAttention(nn.Module):
    """
    Bahdanau-style Attention Mechanism (Paper Section III-D, Equation 6-7)

    Attention score: e(t) = v^T tanh(W_h * h_bi(t) + W_s * s + b)
    Attention weight: alpha(t) = softmax(e(t))
    Context vector: c = sum(alpha(t) * h_bi(t))
    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size

        # W_h: Projects hidden states
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # W_s: Projects context/query vector
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)

        # v: Attention vector
        self.v = nn.Linear(hidden_size, 1, bias=False)

        # Bias term
        self.b = nn.Parameter(torch.zeros(hidden_size))

        # Store attention weights for visualization
        self.weights = None

    def forward(self, lstm_output, query=None):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size) - h_bi(t)
            query: (batch, hidden_size) - context vector s (optional)
                   If None, uses mean of lstm_output as initial context

        Returns:
            context_vector: (batch, hidden_size)
            attention_weights: (batch, seq_len, 1)
        """
        batch_size, seq_len, _ = lstm_output.size()

        # If no query provided, use mean of hidden states
        if query is None:
            query = lstm_output.mean(dim=1)  # (batch, hidden_size)

        # W_h * h_bi(t) for all time steps: (batch, seq_len, hidden_size)
        projected_hidden = self.W_h(lstm_output)

        # W_s * s: (batch, hidden_size) -> (batch, 1, hidden_size)
        projected_query = self.W_s(query).unsqueeze(1)

        # e(t) = v^T * tanh(W_h * h_bi(t) + W_s * s + b)
        # (batch, seq_len, hidden_size)
        energy = torch.tanh(projected_hidden + projected_query + self.b)

        # v^T * energy: (batch, seq_len, 1)
        attention_scores = self.v(energy)

        # alpha(t) = softmax(e(t))
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Store for visualization
        self.weights = attention_weights.detach()

        # c = sum(alpha(t) * h_bi(t)): (batch, hidden_size)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        return context_vector, attention_weights


class LSTMNetwork(nn.Module):
    """
    Bi-LSTM Network with Bahdanau Attention
    Paper Section III-D: Condition-Specific Bi-LSTM Ensemble
    """

    def __init__(self, input_size=5, hidden_size=128, num_layers=3, dropout=0.1, bidirectional=True):
        """
        Args:
            input_size: PHI, R_raw, F, initial-relative R, observed cycle (5 channels)
            hidden_size: LSTM hidden size (Paper Table IV: 128)
            num_layers: Number of LSTM layers (Paper Table IV: 3)
            dropout: Dropout rate (Paper Table IV: 0.1)
            bidirectional: Use bidirectional LSTM (Paper: True)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm_output_size = hidden_size * self.num_directions

        # Bi-LSTM Layer (Paper: 3-layer bidirectional LSTM)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Bahdanau Attention (Paper Equation 6-7)
        self.attention = BahdanauAttention(self.lstm_output_size)

        # Output Head: 256 -> 128 -> 64 -> 1 (Paper Section IV-E)
        self.rul_head = nn.Sequential(
            nn.Linear(self.lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, mc_dropout=False):
        """
        Forward pass

        Args:
            x: (batch, seq_len, input_size)
            mc_dropout: Enable dropout during inference for Monte Carlo Dropout

        Returns:
            output: (batch, 1) - Predicted RUL
        """
        # Bi-LSTM: (batch, seq_len, num_directions * hidden_size)
        lstm_out, _ = self.lstm(x)

        # Bahdanau Attention
        context_vector, attention_weights = self.attention(lstm_out)

        # RUL Prediction
        if mc_dropout:
            # Enable dropout during inference for Monte Carlo Dropout
            self.rul_head.train()

        output = self.rul_head(context_vector)
        return output


class LSTMModel(BaseDeepLearningModel):
    """
    LSTM RUL Prediction Model
    Uses Huber Loss (Paper Section III-E, Equation 11)
    """

    def __init__(self, config):
        super().__init__(config)
        self.build_model()

    def build_model(self):
        """Build model with Huber Loss (Paper: delta = 0.5)"""
        self.model = LSTMNetwork(
            input_size=5,  # PHI, resistance, force, degradation_res, cumulative_cycles
            hidden_size=self.config.LSTM_HIDDEN_SIZE,
            num_layers=self.config.LSTM_NUM_LAYERS,
            dropout=self.config.DROPOUT,
            bidirectional=self.config.LSTM_BIDIRECTIONAL
        ).to(self.device)

        # Huber Loss (Paper Equation 11: delta = 0.5)
        self.criterion = nn.HuberLoss(delta=0.5)

        # AdamW Optimizer (Paper Table IV)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,  # 5e-4
            weight_decay=1e-4
        )

        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.PATIENCE // 2
        )

    def fit(self, train_loader, val_loader=None):
        """Train model"""
        print(f"\n[{self.get_model_name()}] Training started...")
        print("="*70)

        best_val_rmse = float('inf')
        best_epoch = 0
        best_state = None
        patience_counter = 0

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse_norm': [],
            'val_rmse_norm': [],
            'train_rmse_orig': [],
            'val_rmse_orig': [],
            'learning_rates': []
        }

        for epoch in range(self.config.EPOCHS):
            # Training
            train_loss, train_preds, train_targets = self._train_one_epoch(train_loader, epoch)

            # Validation
            if val_loader is not None:
                val_loss, val_preds, val_targets = self._validate(val_loader)
            else:
                val_loss, val_preds, val_targets = 0, train_preds, train_targets

            # Calculate RMSE
            train_rmse_norm = calculate_rmse(train_preds, train_targets)
            val_rmse_norm = calculate_rmse(val_preds, val_targets)

            # Original space RMSE
            train_preds_orig = inverse_transform_rul(train_preds, self.config.MAX_RUL_CYCLE, self.config.MIN_RUL_CYCLE)
            train_targets_orig = inverse_transform_rul(train_targets, self.config.MAX_RUL_CYCLE, self.config.MIN_RUL_CYCLE)
            val_preds_orig = inverse_transform_rul(val_preds, self.config.MAX_RUL_CYCLE, self.config.MIN_RUL_CYCLE)
            val_targets_orig = inverse_transform_rul(val_targets, self.config.MAX_RUL_CYCLE, self.config.MIN_RUL_CYCLE)

            train_rmse_orig = calculate_rmse(train_preds_orig, train_targets_orig)
            val_rmse_orig = calculate_rmse(val_preds_orig, val_targets_orig)

            # Scheduler update
            if self.config.EARLY_STOP_METRIC == 'original':
                self.scheduler.step(val_rmse_orig)
            else:
                self.scheduler.step(val_rmse_norm)

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_rmse_norm'].append(train_rmse_norm)
            history['val_rmse_norm'].append(val_rmse_norm)
            history['train_rmse_orig'].append(train_rmse_orig)
            history['val_rmse_orig'].append(val_rmse_orig)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Best model check
            if val_rmse_orig < best_val_rmse:
                best_val_rmse = val_rmse_orig
                best_epoch = epoch
                self.save(self.config.MODEL_PATH)
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                best_marker = "v"
                patience_counter = 0
            else:
                best_marker = " "
                patience_counter += 1

            # Print progress
            print(f"{best_marker} Epoch {epoch+1:3d}/{self.config.EPOCHS} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"RMSE(N): {train_rmse_norm:.4f}/{val_rmse_norm:.4f} | "
                  f"RMSE(O): {train_rmse_orig:.1f}/{val_rmse_orig:.1f}")

            # Early stopping
            if patience_counter >= self.config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print("="*70)
        print(f"Training completed! Best Epoch: {best_epoch+1}, Best Val RMSE: {best_val_rmse:.1f} cycles")
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def predict_with_uncertainty(self, dataloader, n_iter=30):
        """
        Monte Carlo Dropout for Uncertainty Quantification
        Paper Section III-E, Equation 10
        """
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = batch['inputs'].to(self.device)
                    targets = batch['rul'].cpu().numpy()
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.cpu().numpy().squeeze()

                # Monte Carlo Dropout: T stochastic forward passes
                mc_predictions = []
                for _ in range(n_iter):
                    output = self.model(inputs, mc_dropout=True)
                    mc_predictions.append(output.cpu().numpy())

                mc_predictions = np.array(mc_predictions).squeeze(-1)

                all_predictions.append(mc_predictions)
                all_targets.append(targets)

        all_predictions = np.concatenate(all_predictions, axis=1)
        all_targets = np.concatenate(all_targets)

        # Mean and Std (Equation 10)
        predictions_mean = np.mean(all_predictions, axis=0)
        predictions_std = np.std(all_predictions, axis=0)

        return predictions_mean, predictions_std, all_predictions, all_targets

    def get_model_name(self):
        return "Bi-LSTM + Bahdanau Attention"
