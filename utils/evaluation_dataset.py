"""
Evaluation Dataset for RUL Prediction
IEEE Paper: Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from config.config import config as global_config


class PartialDataEvaluationDataset(Dataset):
    """
    Evaluation Dataset for Final RUL Prediction
    Uses only partial data (observed up to a point) to predict remaining useful life.
    """

    def __init__(self, file_data_list, window_size=96, data_ratio=0.7):
        self.window_size = window_size
        self.samples = []

        for file_data in file_data_list:
            if file_data is None:
                continue

            inputs = file_data['inputs']
            rul_normalized = file_data['rul']
            rul_original = file_data['rul_original']

            # Final RUL at the last observed point
            last_observed_idx = len(inputs) - 1
            final_rul_norm = rul_normalized[last_observed_idx]
            final_rul_orig = rul_original[last_observed_idx]

            if len(inputs) >= window_size:
                input_window = inputs[-window_size:]

                self.samples.append({
                    'x': input_window,
                    'y_norm': final_rul_norm,
                    'y_orig': final_rul_orig,
                    'file_length': len(inputs),
                    'original_total_length': file_data['original_length']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['x']),
            torch.FloatTensor([sample['y_norm']]),
            sample['y_orig']
        )


def evaluate_ensemble_rul_prediction(model_0A, model_0_5A, model_1A, test_data_list, device, config, n_samples=30):
    """
    Ensemble RUL Prediction with Uncertainty Quantification
    Paper Section III-E: MC Dropout & Weighted Ensemble
    """
    from .preprocessing import inverse_transform_rul

    # 1. Set Model Mode (Train mode for MC Dropout)
    if n_samples > 1:
        model_0A.model.train()
        model_0_5A.model.train()
        model_1A.model.train()
    else:
        model_0A.model.eval()
        model_0_5A.model.eval()
        model_1A.model.eval()

    predictions = []
    targets = []
    files_info = []

    with torch.no_grad():
        for file_data in test_data_list:
            inputs = file_data['inputs']
            rul_orig = file_data['rul_original']
            soft_labels = file_data['soft_labels']

            window_size = config.WINDOW_SIZE
            if len(inputs) < window_size:
                continue

            last_window = inputs[-window_size:]
            current_probs = soft_labels.mean(axis=0)

            # Weight Sharpening
            power = getattr(config, 'ENSEMBLE_WEIGHT_POWER', 3.0)
            if power > 1.0:
                current_probs = current_probs ** power
                current_probs = current_probs / (current_probs.sum() + 1e-10)

            last_rul = rul_orig[-1]

            # Add accumulative cycle channel
            cumulative_cycles = file_data['cumulative_contact'][-1]
            cumulative_scaled = np.clip(cumulative_cycles / config.CUMULATIVE_MAX_CYCLE, 0, 1)
            cumulative_channel = np.full((window_size, 1), cumulative_scaled)
            last_window_with_cumulative = np.concatenate([last_window, cumulative_channel], axis=1)

            x = torch.FloatTensor(last_window_with_cumulative).unsqueeze(0).to(device)

            # === MC Sampling ===
            mc_predictions = []
            
            # Temporary lists for individual model predictions
            mc_pred_0A_list = []
            mc_pred_0_5A_list = []
            mc_pred_1A_list = []

            for _ in range(n_samples):
                # Predict with each model
                pred_0A_norm = model_0A.model(x).cpu().numpy()[0, 0]
                pred_0_5A_norm = model_0_5A.model(x).cpu().numpy()[0, 0]
                pred_1A_norm = model_1A.model(x).cpu().numpy()[0, 0]

                pred_0A = inverse_transform_rul(pred_0A_norm, config.MAX_RUL_CYCLE, config.MIN_RUL_CYCLE)
                pred_0_5A = inverse_transform_rul(pred_0_5A_norm, config.MAX_RUL_CYCLE, config.MIN_RUL_CYCLE)
                pred_1A = inverse_transform_rul(pred_1A_norm, config.MAX_RUL_CYCLE, config.MIN_RUL_CYCLE)

                mc_pred_0A_list.append(pred_0A)
                mc_pred_0_5A_list.append(pred_0_5A)
                mc_pred_1A_list.append(pred_1A)

                # Weighted Ensemble
                w_0A, w_0_5A, w_1A = current_probs[0], current_probs[1], current_probs[2]
                ensemble_pred = w_0A * pred_0A + w_0_5A * pred_0_5A + w_1A * pred_1A
                mc_predictions.append(ensemble_pred)

            mc_predictions = np.array(mc_predictions)
            final_pred = np.mean(mc_predictions)
            uncertainty = np.std(mc_predictions)

            predictions.append(final_pred)
            targets.append(last_rul)

            files_info.append({
                'file_path': file_data.get('file_path'),
                'end_index': file_data.get('end_index', len(inputs)-1),
                'observed_cycle': float(cumulative_cycles),
                'observed_rows': len(inputs),
                'total_rows': file_data['original_length'],
                'cut_fraction_rows': len(inputs)/file_data['original_length'],
                'rul_source': file_data.get('rul_source', 'provided_column'),
                'pred_0A': float(np.mean(mc_pred_0A_list)),
                'pred_0.5A': float(np.mean(mc_pred_0_5A_list)),
                'pred_1A': float(np.mean(mc_pred_1A_list)),
                'weights': [float(w_0A), float(w_0_5A), float(w_1A)],
                'ensemble_pred': final_pred,
                'uncertainty': uncertainty,
                'target': last_rul,
                'mc_predictions': mc_predictions.tolist()
            })

    # Restore Eval mode
    model_0A.model.eval()
    model_0_5A.model.eval()
    model_1A.model.eval()

    return {
        'predictions': np.array(predictions),
        'targets': np.array(targets),
        'files_info': files_info
    }
