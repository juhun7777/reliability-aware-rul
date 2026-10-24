"""
Configuration Management Module
IEEE Paper: Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins
"""
import os
import random
import numpy as np
import torch


class Config:
    """Global Configuration Class (Matched with Paper Table IV)"""

    # Seed
    SEED = 42

    # Data Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT = os.path.join(BASE_DIR, 'data')
    FOLDERS = {
        '0A': 'train/0A',
        '0.5A': 'train/0.5A',
        '1.0A': 'train/1.0A'
    }

    # RUL Normalization (Min-Max + Log Normalization)
    MIN_RUL_CYCLE = 0.0
    MAX_RUL_CYCLE = 380000.0

    # Cumulative Cycle Normalization
    CUMULATIVE_MAX_CYCLE = 380000.0

    # Data Split Configuration (Adapted for minimal sample dataset)
    N_TRAIN = 1
    N_VAL = 1
    N_TEST = 1
    TEST_CUT_RANGE = (0.8, 0.9)
    TEST_OBSERVATION_FRACTION = 0.8  # fixed revised evaluation, not historical random cuts

    # Time-series Window (Paper Table IV)
    WINDOW_SIZE = 200  # Input window size
    STRIDE = 10        # Stride for data augmentation

    # PHI Parameters (Paper Equation 1, Algorithm 1)
    # R_ideal(F) = alpha * F^(-1/3) + beta
    ALPHA = 2.34    # Paper: alpha = 2.34
    BETA = 0.133    # Paper: beta = 0.133 (FIXED from 0.087)
    PHI_MAX_RESIDUAL = 30.0  # Global threshold for PHI normalization

    # Transformer Model Hyperparameters
    D_MODEL = 128
    N_HEADS = 4
    E_LAYERS = 2
    D_LAYERS = 1
    D_FF = 256
    DROPOUT = 0.1  # Paper Table IV: 0.1

    # LSTM Model Hyperparameters (Paper Table IV)
    LSTM_HIDDEN_SIZE = 128      # Paper: 128 (x2 BiDir)
    LSTM_NUM_LAYERS = 3         # Paper: 3 layers
    LSTM_BIDIRECTIONAL = True   # Paper: Bidirectional

    # Training Hyperparameters (Paper Table IV)
    BATCH_SIZE = 32             # Paper: 32
    EPOCHS = 50                 # Paper: 50 (max epochs)
    LEARNING_RATE = 0.0005      # Paper: 5x10^-4
    PATIENCE = 15               # Paper: Early stopping = 15

    # Results Path
    SAVE_DIR = './results'
    MODEL_PATH = './results/best_model.pth'

    # Early Stopping Metric
    EARLY_STOP_METRIC = 'original'

    # XAI Analysis
    ENABLE_XAI = True

    # Soft Label Method
    SOFT_LABEL_METHOD = 'STFT'  # 'STFT' or 'RF'

    # RF (Random Forest) Parameters
    RF_WINDOW_SIZE = 400
    ENSEMBLE_WEIGHT_POWER = 3.0

    # STFT Parameters (Paper Section III-C, Algorithm 2)
    STFT_WINDOW_SIZE = 400      # Paper Table IV: 400
    STFT_STRIDE = 10
    STFT_FS = 100               # Paper: fs = 100Hz
    STFT_NPERSEG = 256          # Paper: n_perseg = 256
    STFT_NOVERLAP = 128         # Paper: n_overlap = 128
    STFT_MLP_EPOCHS = 150       # Paper: MLP training epochs
    STFT_MLP_BATCH_SIZE = 128
    STFT_MLP_LR = 0.001
    STFT_EARLY_STOP_PATIENCE = 10
    STFT_MLP_DROPOUT = 0.3      # Paper: p = 0.3

    # Machine Learning Model Configuration
    ML_FEATURES_WINDOW = 200
    ML_FEATURES_STRIDE = 25

    # XGBoost Parameters
    XGBOOST_PARAMS = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds': 50,
        'random_state': SEED,
        'n_jobs': -1
    }

    # LightGBM Parameters
    LIGHTGBM_PARAMS = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds': 50,
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1
    }

    # CatBoost Parameters
    CATBOOST_PARAMS = {
        'iterations': 300,
        'depth': 4,
        'learning_rate': 0.05,
        'min_data_in_leaf': 5,
        'l2_leaf_reg': 3.0,
        'early_stopping_rounds': 50,
        'random_seed': SEED,
        'verbose': False
    }

    @classmethod
    def set_seed(cls):
        """Set seed for reproducibility"""
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_device(cls):
        """Return available device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(f'{cls.SAVE_DIR}/xai_analysis', exist_ok=True)
        os.makedirs(f'{cls.SAVE_DIR}/models', exist_ok=True)


# Global config instance
config = Config()
