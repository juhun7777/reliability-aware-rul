"""
PyTorch Dataset Module
IEEE Paper: Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from config.config import config


class RULDataset(Dataset):
    """Time-series Window-based RUL Dataset"""

    def __init__(self, file_data_list, window_size=96, stride=10):
        """
        Args:
            file_data_list: List of data loaded by load_single_file()
            window_size: Input window size
            stride: Sliding window stride
        """
        self.window_size = window_size
        self.stride = stride
        self.samples = []

        for file_data in file_data_list:
            if file_data is None:
                continue

            inputs = file_data['inputs']
            targets = file_data['rul']
            targets_original = file_data['rul_original']

            # Sliding window (Start from 60% of file to match TEST distribution roughly)
            start_idx = int(len(inputs) * 0.6)
            for i in range(start_idx, len(inputs) - window_size + 1, stride):
                window = inputs[i:i + window_size]  # (window_size, n_features)

                # Add cumulative cycle count channel
                window_end_idx = i + window_size - 1
                cumulative_cycles = file_data['cumulative_contact'][window_end_idx]
                
                # Global Max Normalization
                cumulative_scaled = np.clip(cumulative_cycles / config.CUMULATIVE_MAX_CYCLE, 0, 1)

                # Add channel
                cumulative_channel = np.full((window_size, 1), cumulative_scaled)
                window_with_cumulative = np.concatenate([window, cumulative_channel], axis=1)

                self.samples.append({
                    'x': window_with_cumulative,  # (window_size, n_features+1)
                    'y': targets[i + window_size - 1]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.FloatTensor(sample['x']), torch.FloatTensor([sample['y']])
