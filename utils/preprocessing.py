"""
Data Preprocessing Module
IEEE Paper: Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins

PHI Calculation (Section III-B, Algorithm 1)
STFT-based Soft Labeling (Section III-C, Algorithm 2)
RUL Normalization
"""
import numpy as np
import pandas as pd
from utils.csv_io import read_csv as read_sensor_csv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from scipy.signal import stft
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


def inverse_transform_rul(normalized_rul, max_cycle=400000.0, min_cycle=1000.0):
    """
    Inverse transform Min-Max Log-normalized RUL to original cycle values

    Args:
        normalized_rul: Normalized RUL in [0, 1] range
        max_cycle: Maximum cycle for normalization
        min_cycle: Minimum cycle for normalization

    Returns:
        Original RUL in cycles
    """
    min_log = np.log1p(min_cycle)
    max_log = np.log1p(max_cycle)
    rul_log = normalized_rul * (max_log - min_log) + min_log
    return np.expm1(rul_log)


class PHICalculator:
    """
    Physics-based Health Indicator (PHI) Calculator
    Paper Section III-B, Algorithm 1, Equation 1-2

    R_ideal(F) = alpha * F^(-1/3) + beta
    Residual(t) = R_measured(t) - R_ideal(F(t))
    Revised PHI = clip((abs(Residual) - initial100mean) / theta_max, 0, 1).
    legacy_full_min is an explicit historical replay option.
    """

    def __init__(self, alpha=2.34, beta=0.133, max_residual_threshold=30.0,
                 baseline_rows=100, baseline_method='initial_mean'):
        """
        Args:
            alpha: Scaling coefficient (Paper: 2.34)
            beta: Baseline offset resistance (Paper: 0.133)
            max_residual_threshold: Global threshold for PHI normalization
        """
        self.alpha = alpha
        self.beta = beta  # Paper value: 0.133
        self.max_residual_threshold = max_residual_threshold
        self.baseline_rows = baseline_rows
        if baseline_method not in ('initial_mean', 'legacy_full_min'):
            raise ValueError('baseline_method must be initial_mean or legacy_full_min')
        self.baseline_method = baseline_method

    def calculate_phi(self, resistance, force):
        """
        Calculate PHI from resistance and force data

        Args:
            resistance: Resistance data array
            force: Force data array

        Returns:
            phi: PHI values [0, 1]
            R: Resistance values
            R_theoretical: Theoretical resistance values
        """
        R = np.array(resistance)
        F = np.array(force)

        # Prevent division by zero
        F = np.where(F <= 0, 1e-10, F)

        # Theoretical resistance (Equation 1)
        # R_ideal(F) = alpha * F^(-1/3) + beta
        R_theoretical = self.alpha * (F ** (-1/3)) + self.beta

        # Residual calculation (Equation 2)
        residual = np.abs(R - R_theoretical)

        # Global Normalization
        # Initial calibration uses observed samples only. Full-trajectory minimum
        # is available explicitly for historical replay, not the revised protocol.
        baseline_residual = (residual.min() if self.baseline_method == 'legacy_full_min'
                             else residual[:self.baseline_rows].mean())
        phi = (residual - baseline_residual) / self.max_residual_threshold

        # Clip to [0, 1]
        phi = np.clip(phi, 0, 1)

        return phi, R, R_theoretical


class SoftLabelGenerator:
    """Random Forest based Soft Label Generator (Supervised)"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_map = {'0A': 0, '0.5A': 1, '1.0A': 2}

    def extract_statistical_features(self, resistance_window):
        """Extract statistical features from resistance window"""
        features = []

        # Basic statistics
        features.extend([
            np.mean(resistance_window),
            np.std(resistance_window),
            np.max(resistance_window),
            np.min(resistance_window),
            np.median(resistance_window),
            np.percentile(resistance_window, 25),
            np.percentile(resistance_window, 75),
            np.ptp(resistance_window)
        ])

        # Difference statistics
        diff = np.diff(resistance_window)
        features.extend([
            np.mean(diff),
            np.std(diff),
            np.max(diff),
            np.min(diff),
            np.median(diff),
            np.mean(np.abs(diff)),
            np.sum(diff > 0) / len(diff),
            np.sum(diff < 0) / len(diff)
        ])

        # Advanced statistics
        features.extend([
            stats.skew(resistance_window),
            stats.kurtosis(resistance_window),
            np.std(resistance_window) / (np.mean(resistance_window) + 1e-10),
            np.sqrt(np.mean(resistance_window**2))
        ])

        # Trend features
        x = np.arange(len(resistance_window))
        slope, _ = np.polyfit(x, resistance_window, 1)
        features.extend([
            slope,
            resistance_window[-1] - resistance_window[0]
        ])

        return np.array(features)

    def train_classifier(self, train_data_dict, window_size=96, stride=10):
        """Train Random Forest classifier (Supervised)"""
        print("[Soft Label] Training Random Forest Classifier...")
        all_features = []
        all_labels = []

        rf_window_size = min(window_size, 1000)
        rf_stride = max(stride // 5, 20)

        print(f"  Using RF window_size={rf_window_size}, stride={rf_stride}")

        for current, file_paths in train_data_dict.items():
            if current not in self.label_map:
                continue

            label = self.label_map[current]
            print(f"  Processing {current} ({len(file_paths)} files)...")

            for file_path in tqdm(file_paths, desc=f"  Extracting {current}", leave=False):
                try:
                    df = read_sensor_csv(file_path)
                    df.columns = df.columns.str.replace('\ufeff', '').str.strip()
                    if 'res' not in df.columns:
                        continue

                    if len(df) < rf_window_size:
                        continue

                    resistance = df['res'].values

                    for i in range(0, len(resistance) - rf_window_size, rf_stride):
                        window = resistance[i:i + rf_window_size]
                        features = self.extract_statistical_features(window)
                        all_features.append(features)
                        all_labels.append(label)
                except:
                    continue

        X = np.array(all_features)
        y = np.array(all_labels)
        print(f"  Extracted {len(X)} feature samples")
        print(f"  Class distribution: {Counter(y)}")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)

        train_acc = self.model.score(X_scaled, y)
        print(f"  Random Forest trained. Training Accuracy: {train_acc*100:.2f}%")

    def generate_soft_labels(self, resistance_data, window_size=96):
        """Generate Soft Labels (predict_proba)"""
        if self.model is None or self.scaler is None:
            return np.ones((len(resistance_data), 3)) / 3

        features_list = []
        valid_indices = []

        for i in range(len(resistance_data)):
            start = max(0, i - window_size // 2)
            end = min(len(resistance_data), i + window_size // 2)
            window = resistance_data[start:end]

            if len(window) < 10:
                continue

            features = self.extract_statistical_features(window)
            features_list.append(features)
            valid_indices.append(i)

        if not features_list:
            return np.ones((len(resistance_data), 3)) / 3

        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)

        result_labels = np.ones((len(resistance_data), 3)) / 3
        result_labels[valid_indices] = probs

        return result_labels


class DeepMLP_Logits(nn.Module):
    """
    Deep MLP for STFT-based soft label generation
    Paper Section III-C, Figure 2: 160 -> 256 -> 128 -> 64 -> 3
    """

    def __init__(self, input_dim=160, hidden_dims=[256, 128, 64], output_dim=3, dropout=0.3):
        super(DeepMLP_Logits, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class STFTSoftLabelGenerator:
    """
    STFT + Deep MLP based Soft Label Generator
    Paper Section III-C, Algorithm 2
    """

    def __init__(self, fs=100, nperseg=256, noverlap=128):
        """
        Args:
            fs: Sampling frequency (Paper: 100Hz)
            nperseg: STFT segment size (Paper: 256)
            noverlap: STFT overlap size (Paper: 128)
        """
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.model = None
        self.scaler = None
        self.trained_window_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def extract_hybrid_features_window(self, window_data):
        """
        Extract STFT frequency features + time domain features
        Paper Algorithm 2: 128-dim frequency + 32-dim time = 160-dim hybrid

        Returns:
            features: 160-dimensional feature vector
        """
        # STFT calculation
        f, t, Zxx = stft(window_data, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
        Sxx = np.abs(Zxx)

        # Frequency features (128 dimensions)
        freq_mean = np.mean(Sxx, axis=1)
        freq_std = np.std(Sxx, axis=1)
        freq_max = np.max(Sxx, axis=1)
        freq_min = np.min(Sxx, axis=1)

        freq_features_raw = np.concatenate([freq_mean, freq_std, freq_max, freq_min])

        # Interpolate to 128 dimensions if needed
        if len(freq_features_raw) != 128:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(freq_features_raw))
            x_new = np.linspace(0, 1, 128)
            f_interp = interp1d(x_old, freq_features_raw, kind='linear')
            freq_features = f_interp(x_new)
        else:
            freq_features = freq_features_raw

        # Time domain statistical features (32 dimensions)
        time_features = []

        # Basic statistics (8)
        time_features.extend([
            np.mean(window_data),
            np.std(window_data),
            np.max(window_data),
            np.min(window_data),
            np.median(window_data),
            np.percentile(window_data, 25),
            np.percentile(window_data, 75),
            np.ptp(window_data)
        ])

        # Difference statistics (8)
        diff = np.diff(window_data)
        time_features.extend([
            np.mean(diff),
            np.std(diff),
            np.max(diff),
            np.min(diff),
            np.median(diff),
            np.mean(np.abs(diff)),
            np.sum(diff > 0) / max(len(diff), 1),
            np.sum(diff < 0) / max(len(diff), 1)
        ])

        # Advanced statistics (8)
        time_features.extend([
            stats.skew(window_data),
            stats.kurtosis(window_data),
            np.std(window_data) / (np.mean(window_data) + 1e-10),
            np.sqrt(np.mean(window_data**2)),
            np.sum(window_data > np.mean(window_data)) / len(window_data),
            np.sum(window_data < np.mean(window_data)) / len(window_data),
            (np.max(window_data) - np.min(window_data)) / (np.mean(window_data) + 1e-10),
            np.var(window_data)
        ])

        # Trend features (8)
        x = np.arange(len(window_data))
        slope, intercept = np.polyfit(x, window_data, 1)
        time_features.extend([
            slope,
            intercept,
            window_data[-1] - window_data[0],
            np.mean(window_data[:len(window_data)//2]) - np.mean(window_data[len(window_data)//2:]),
            np.std(window_data[:len(window_data)//2]),
            np.std(window_data[len(window_data)//2:]),
            np.corrcoef(x, window_data)[0, 1],
            np.max(window_data) - np.min(window_data)
        ])

        time_features = np.array(time_features)

        # Weighted combination (freq: 65%, time: 35%)
        freq_weighted = freq_features * 0.65
        time_weighted = time_features * 0.35

        hybrid_features = np.concatenate([freq_weighted, time_weighted])
        return np.nan_to_num(hybrid_features, nan=0., posinf=0., neginf=0.)

    def train_mlp(self, train_data_dict, window_size=400, stride=10, epochs=150,
                  batch_size=128, lr=0.001, patience=10, dropout=0.3,
                  validation_data_dict=None, group_map=None, seed=42):
        """
        Train Deep MLP (Paper Section III-C)

        Args:
            train_data_dict: File path dictionary
            window_size: Window size (Paper Table IV: 400)
            stride: Stride
            epochs: Max training epochs
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
            dropout: Dropout rate (Paper: 0.3)
        """
        print("[STFT Soft Label] Training Deep MLP...")
        all_features = []
        all_labels = []
        all_groups = []
        validation_flags = []
        label_map = {'0A': 0, '0.5A': 1, '1.0A': 2}

        print(f"  Using STFT window_size={window_size}, stride={stride}")

        sources = [(c, paths, False) for c, paths in train_data_dict.items()]
        if validation_data_dict is not None:
            sources += [(c, paths, True) for c, paths in validation_data_dict.items()]
        for current, file_paths, is_validation in sources:
            label = label_map[current]
            for file_path in tqdm(file_paths, desc=f"  Extracting {current}", leave=False):
                try:
                    df = read_sensor_csv(file_path)
                    df.columns = df.columns.str.replace('\ufeff', '').str.strip()
                    if 'res' not in df.columns:
                        continue
                    if len(df) < window_size:
                        continue

                    resistance = df['res'].values
                    for i in range(0, len(resistance) - window_size + 1, stride):
                        window = resistance[i:i + window_size]
                        features = self.extract_hybrid_features_window(window)
                        all_features.append(features)
                        all_labels.append(label)
                        key = str(Path(file_path).resolve())
                        all_groups.append((group_map or {}).get(key, key))
                        validation_flags.append(is_validation)
                except Exception as e:
                    continue

        X = np.array(all_features)
        y = np.array(all_labels)
        print(f"  Extracted {len(X)} feature samples")

        if len(X) == 0:
            raise ValueError('No gate features extracted; inspect CSV columns, encodings and lengths.')

        print(f"  Class distribution: {Counter(y)}")

        groups = np.asarray(all_groups)
        if validation_data_dict is not None:
            val_idx = np.flatnonzero(validation_flags)
            train_idx = np.flatnonzero(~np.asarray(validation_flags))
        else:
            # Keep every window from a file/parent on the same side. A supplied
            # group_map also keeps augmented siblings together.
            splits = GroupShuffleSplit(n_splits=50, test_size=.2, random_state=seed)
            valid = [(a, b) for a, b in splits.split(X, y, groups)
                     if len(np.unique(y[a])) == 3 and len(np.unique(y[b])) == 3]
            if not valid:
                raise ValueError('Need distinct groups for all three classes; provide validation_data_dict.')
            train_idx, val_idx = valid[0]
        if not len(train_idx) or not len(val_idx):
            raise ValueError('Nonempty training and validation groups are required.')
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise ValueError('STFT training and validation groups overlap.')
        if set(y[train_idx]) != {0, 1, 2} or set(y[val_idx]) != {0, 1, 2}:
            raise ValueError('All three source classes must appear in each gate split.')
        self.scaler = StandardScaler().fit(X[train_idx])
        X_train, X_val = self.scaler.transform(X[train_idx]), self.scaler.transform(X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]
        self.validation_protocol = dict(
            split='explicit_groups' if validation_data_dict is not None else 'group_holdout',
            train_groups=sorted(set(groups[train_idx])),
            validation_groups=sorted(set(groups[val_idx])),
            scaler_fit='training_only', seed=seed)

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

        # PyTorch datasets
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        if len(train_dataset) < 2 or batch_size < 2:
            raise ValueError('STFT BatchNorm training needs at least two examples per batch.')
        train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=min(batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Model setup (Paper Figure 2: 160->256->128->64->3)
        self.model = DeepMLP_Logits(input_dim=160, hidden_dims=[256, 128, 64], output_dim=3, dropout=dropout)
        self.model = self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        # Training loop with Early Stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        final_accuracy = 0.0

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_acc = 100 * correct / total if total > 0 else 0
            final_accuracy = val_acc

            scheduler.step(avg_val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = {k: v.detach().cpu().clone()
                                    for k, v in self.model.state_dict().items()}
                best_val_accuracy = val_acc
                best_epoch = epoch + 1
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early Stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
                    break

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print("  Loaded best model weights.")
            self.validation_protocol.update(best_epoch=best_epoch,
                                            best_val_accuracy=best_val_accuracy)

        self.trained_window_size = window_size
        print(f"  MLP trained. Best Val Loss: {best_val_loss:.4f}, Final Val Acc: {final_accuracy:.2f}%")

    def generate_soft_labels(self, resistance_data, window_size=None):
        """
        Generate Soft Labels using trained MLP
        Paper Equation 5: w_c = softmax(z_c)
        """
        if self.model is None or self.scaler is None:
            return np.ones((len(resistance_data), 3)) / 3

        if self.trained_window_size is not None:
            window_size = self.trained_window_size
        elif window_size is None:
            window_size = 400

        self.model.eval()
        soft_labels = []

        with torch.no_grad():
            for i in range(len(resistance_data)):
                start = max(0, i - window_size // 2)
                end = min(len(resistance_data), i + window_size // 2)
                window = resistance_data[start:end]

                if len(window) < 10:
                    soft_labels.append([1/3, 1/3, 1/3])
                    continue

                features = self.extract_hybrid_features_window(window)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                features_tensor = torch.FloatTensor(features_scaled).to(self.device)

                logits = self.model(features_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                soft_labels.append(probs)

        return np.array(soft_labels)


class RULNormalizer:
    """RUL Normalization/Denormalization (Min-Max Log Normalization)"""

    def __init__(self, max_rul_cycle=400000.0, min_rul_cycle=1000.0):
        self.max_rul_cycle = max_rul_cycle
        self.min_rul_cycle = min_rul_cycle
        self.min_log = np.log1p(min_rul_cycle)
        self.max_log = np.log1p(max_rul_cycle)
        self.log_range = self.max_log - self.min_log

    def normalize(self, rul):
        """Normalize RUL to [0, 1] range using Min-Max Log normalization"""
        rul_log = np.log1p(rul)
        rul_normalized = (rul_log - self.min_log) / self.log_range
        return np.clip(rul_normalized, 0.0, 1.0)

    def inverse_transform(self, normalized_rul):
        """Inverse transform normalized RUL to original cycle values"""
        rul_log = normalized_rul * self.log_range + self.min_log
        return np.expm1(rul_log)
