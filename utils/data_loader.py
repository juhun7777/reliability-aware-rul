"""
Data Loading and Splitting Module
IEEE Paper: Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins
"""
import os
import glob
import random
import numpy as np
import pandas as pd
from utils.csv_io import read_csv as read_sensor_csv
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils.preprocessing import PHICalculator, SoftLabelGenerator, STFTSoftLabelGenerator, RULNormalizer


class DataLoaderFileBase:
    """File-based Data Loader"""

    def __init__(self, config):
        self.config = config
        self.phi_calculator = PHICalculator(
            alpha=config.ALPHA, 
            beta=config.BETA,
            max_residual_threshold=getattr(config, 'PHI_MAX_RESIDUAL', 30.0)
        )

        if config.SOFT_LABEL_METHOD == 'STFT':
            self.soft_label_generator = STFTSoftLabelGenerator(
                fs=config.STFT_FS,
                nperseg=config.STFT_NPERSEG,
                noverlap=config.STFT_NOVERLAP
            )
        else:
            self.soft_label_generator = SoftLabelGenerator()

        self.rul_normalizer = RULNormalizer(
            max_rul_cycle=config.MAX_RUL_CYCLE,
            min_rul_cycle=config.MIN_RUL_CYCLE
        )
        self.input_scaler = StandardScaler()

        self.train_files = {'0A': [], '0.5A': [], '1.0A': []}
        self.val_files = {'0A': [], '0.5A': [], '1.0A': []}
        self.test_files = {'0A': [], '0.5A': [], '1.0A': []}

    def load_and_split_files(self):
        """Load file paths and split into Train/Val/Test"""
        print("="*70)
        print("[Data Loader] Loading and splitting files...")
        print("="*70)

        for current, folder_name in self.config.FOLDERS.items():
            folder_path = os.path.join(self.config.DATA_ROOT, folder_name)
            if not os.path.exists(folder_path):
                print(f"  Warning: Folder not found: {folder_path}")
                continue

            all_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
            total_needed = self.config.N_TRAIN + self.config.N_VAL + self.config.N_TEST

            if len(all_files) < total_needed:
                print(f"  {current}: Not enough files (Found {len(all_files)}, Need {total_needed})")
                continue

            random.Random(self.config.SEED).shuffle(all_files)

            self.train_files[current] = all_files[:self.config.N_TRAIN]
            self.val_files[current] = all_files[self.config.N_TRAIN:self.config.N_TRAIN + self.config.N_VAL]
            self.test_files[current] = all_files[self.config.N_TRAIN + self.config.N_VAL:total_needed]

            print(f"  {current}: Train={len(self.train_files[current])}, Val={len(self.val_files[current])}, Test={len(self.test_files[current])}")

    def train_soft_label_generator(self):
        """Train Soft Label Generator (STFT+MLP)"""
        if self.config.SOFT_LABEL_METHOD == 'STFT':
            print(f"\n[Data Loader] Training STFT+MLP Soft Label Generator...")
            self.soft_label_generator.train_mlp(
                self.train_files,
                window_size=self.config.STFT_WINDOW_SIZE,
                stride=self.config.STFT_STRIDE,
                epochs=self.config.STFT_MLP_EPOCHS,
                batch_size=self.config.STFT_MLP_BATCH_SIZE,
                lr=self.config.STFT_MLP_LR,
                patience=getattr(self.config, 'STFT_EARLY_STOP_PATIENCE', 10),
                dropout=getattr(self.config, 'STFT_MLP_DROPOUT', 0.3),
                validation_data_dict=self.val_files,
                group_map=getattr(self.config, 'STFT_GROUP_MAP', None),
                seed=self.config.SEED
            )

    def load_single_file(self, file_path, mode='train', fit_scaler=True, current_label=None):
        """Load and preprocess a single file"""
        try:
            df = read_sensor_csv(file_path)
            df.columns = df.columns.str.replace('\ufeff', '').str.strip()

            if 'res' not in df.columns or 'force' not in df.columns:
                return None

            if 'cumulative_contact' not in df.columns:
                raise ValueError('cumulative_contact is required; row counts are not cycle units.')
            rul_source = 'provided_column'
            if 'RUL' not in df.columns:
                df['RUL'] = df['cumulative_contact'].iloc[-1] - df['cumulative_contact']
                rul_source = 'record_end_minus_cycle_not_authenticated_failure'

            if len(df) < self.config.WINDOW_SIZE:
                return None

            original_length = len(df)

            # Test mode: Cut data (80-90%)
            if mode == 'test':
                cut_ratio = getattr(self.config, 'TEST_OBSERVATION_FRACTION', .8)
                cut_idx = int(original_length * cut_ratio)
                cut_idx = max(cut_idx, self.config.WINDOW_SIZE + 10)
                df = df.iloc[:cut_idx]

            resistance = df['res'].values
            force = df['force'].values
            rul = df['RUL'].values

            # Calculate degradation (initial state baseline)
            baseline_window = min(100, len(resistance) // 20)
            baseline_window = max(10, baseline_window)
            baseline_res = resistance[:baseline_window].mean()
            degradation_res = (resistance - baseline_res) / (baseline_res + 1e-10)

            # Normalize RUL
            rul_normalized = self.rul_normalizer.normalize(rul)

            # Calculate PHI
            phi, _, _ = self.phi_calculator.calculate_phi(resistance, force)

            # Generate Soft Labels
            rf_window = getattr(self.config, 'RF_WINDOW_SIZE', 400) if self.config.SOFT_LABEL_METHOD == 'RF' else self.config.STFT_WINDOW_SIZE
            soft_labels = self.soft_label_generator.generate_soft_labels(
                resistance, window_size=rf_window
            )

            # Input Features: [phi, resistance, force, degradation_res]
            inputs = np.stack([phi, resistance, force, degradation_res], axis=1)

            return {
                'inputs': inputs,
                'rul': rul_normalized,
                'rul_original': rul,
                'resistance': resistance,
                'cumulative_contact': df['cumulative_contact'].to_numpy(),
                'file_path': os.path.abspath(file_path),
                'end_index': len(df) - 1,
                'rul_source': rul_source,
                'soft_labels': soft_labels,
                'original_length': original_length,
                'current_label': current_label
            }

        except Exception as e:
            print(f"  Error loading {os.path.basename(file_path)}: {e}")
            return None

    def load_dataset(self, split='train', current_filter=None, fit_scaler=False):
        """Load all files for a split"""
        if split == 'train': files = self.train_files
        elif split == 'val': files = self.val_files
        elif split == 'test': files = self.test_files
        else: raise ValueError(f"Invalid split: {split}")

        data_list = []
        currents = [current_filter] if current_filter else self.config.FOLDERS.keys()

        for current in currents:
            for file_path in tqdm(files[current], desc=f"  {current} {split.capitalize()}", leave=False):
                file_data = self.load_single_file(file_path, mode=split, fit_scaler=fit_scaler, current_label=current)
                if file_data is not None:
                    data_list.append(file_data)
        
        return data_list
