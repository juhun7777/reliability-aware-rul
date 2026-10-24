"""
Main Training Script
IEEE Paper: Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins

Runs the proposed framework:
1. PHI Calculation (Physics-Informed)
2. STFT-based Soft Labeling
3. Condition-Specific Bi-LSTM Ensemble Training
4. Uncertainty-Aware Evaluation (MC Dropout)
"""
import argparse
import json
from pathlib import Path
import hashlib
import pandas as pd
import joblib
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt

from config.config import config
from utils.data_loader import DataLoaderFileBase
from utils.dataset import RULDataset
from models.ensemble_lstm import EnsembleLSTMModel
from models.lstm import LSTMModel

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def create_dataloaders(train_dataset, val_dataset, batch_size=32):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def train_ensemble_models(data_loader, config):
    """
    Train 3 Condition-Specific LSTM Models (0A, 0.5A, 1.0A)
    """
    print(f"\n{'='*70}")
    print(f"Training Ensemble: 3 Specialist Bi-LSTM Models")
    print(f"{'='*70}")

    models = {}
    histories = {}

    # Load ALL training data for global scaler fitting (unused effectively as inputs are raw)
    # But kept for consistency if any scaler logic exists
    _ = data_loader.load_dataset('train', fit_scaler=True) 

    # Load Test Data
    test_data_list = data_loader.load_dataset('test', fit_scaler=False)

    # Train each condition specific model
    for current_name in ['0A', '0.5A', '1.0A']:
        print(f"\n{'─'*70}")
        print(f"Training Specialist Model for {current_name}")
        print(f"{'─'*70}")

        # Load data for specific current
        train_data_list = data_loader.load_dataset('train', current_filter=current_name, fit_scaler=False)
        val_data_list = data_loader.load_dataset('val', current_filter=current_name, fit_scaler=False)

        # Create Datasets
        train_dataset = RULDataset(train_data_list, config.WINDOW_SIZE, config.STRIDE)
        val_dataset = RULDataset(val_data_list, config.WINDOW_SIZE, config.STRIDE)

        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=config.BATCH_SIZE)

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        # Create and Train Model
        model = LSTMModel(config)
        history = model.fit(train_loader, val_loader)

        # Save Model
        model_save_path = config.MODEL_PATH.replace('.pth', f'_{current_name}.pth')
        model.save(model_save_path)
        print(f"  ✅ Model saved: {model_save_path}")

        models[current_name] = model
        histories[current_name] = history

    return models, histories, test_data_list


def evaluate_framework(models, test_data_list, config):
    """
    Evaluate the full framework using Soft-Weighted Ensemble and MC Dropout
    """
    print(f"\n{'='*70}")
    print(f"Ensemble Evaluation: Soft-weighted Voting + Uncertainty Estimation (MC Dropout)")
    print(f"{'='*70}")

    from utils.evaluation_dataset import evaluate_ensemble_rul_prediction
    from utils.metrics import calculate_rmse, calculate_mae, calculate_nasa_score, calculate_mape

    # MC Dropout Evaluation (n_samples=30 for uncertainty quantification)
    ensemble_results = evaluate_ensemble_rul_prediction(
        models['0A'], models['0.5A'], models['1.0A'],
        test_data_list, config.get_device(), config,
        n_samples=30 
    )

    predictions = ensemble_results['predictions']
    targets = ensemble_results['targets']
    files_info = ensemble_results['files_info']
    # Preserve observation identity, all expert means/weights and actual MC draws.
    output = Path(config.SAVE_DIR)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'file_predictions.json').open('w', encoding='utf-8') as f:
        json.dump(files_info, f, indent=2, default=lambda x: x.item())
    pd.DataFrame([{k: v for k, v in r.items() if k != 'mc_predictions'}
                  for r in files_info]).to_csv(output/'file_predictions.csv', index=False)
    np.save(output/'mc_predictions.npy', np.array([r['mc_predictions'] for r in files_info]))

    # Calculate Metric
    metrics = {
        'rmse': calculate_rmse(predictions, targets),
        'mae': calculate_mae(predictions, targets),
        'mape': calculate_mape(predictions, targets),
        'nasa_score': calculate_nasa_score(predictions, targets)
    }

    # Print Results
    print(f"\n📊 Final Results (Proposed Framework):")
    print(f"  RMSE:       {metrics['rmse']:>10.2f} cycles")
    print(f"  MAE:        {metrics['mae']:>10.2f} cycles")
    print(f"  MAPE:       {metrics['mape']:>10.2f}%")
    print(f"  NASA Score: {metrics['nasa_score']:>10.2f}")

    # Detailed File Results Table
    print(f"\n{'='*70}")
    print(f"📋 Test File Details (Ensemble Prediction with Uncertainty):")
    print(f"{'='*70}")
    print(f"\n  {'File':<6} {'Pred 0A':>9} {'w0A':<6} {'Pred 0.5A':>11} {'w0.5A':<7} {'Pred 1A':>9} {'w1A':<6} {'Ensemble (Mean ± Std)':>22} {'True':>9} {'Error':>9} {'Err%':>7}")
    print(f"  {'-'*6} {'-'*9} {'-'*6} {'-'*11} {'-'*7} {'-'*9} {'-'*6} {'-'*22} {'-'*9} {'-'*9} {'-'*7}")

    for i, f in enumerate(files_info):
        ensemble_str = f"{f['ensemble_pred']:>8.1f} ± {f['uncertainty']:<5.1f}"
        error = f['ensemble_pred'] - f['target']
        per_err = abs(error) / f['target'] * 100
        
        print(f"  {i+1:<6} "
              f"{f['pred_0A']:>9.1f} "
              f"{f['weights'][0]:<6.3f} "
              f"{f['pred_0.5A']:>11.1f} "
              f"{f['weights'][1]:<7.3f} "
              f"{f['pred_1A']:>9.1f} "
              f"{f['weights'][2]:<6.3f} "
              f"{ensemble_str:>22} "
              f"{f['target']:>9.1f} "
              f"{error:>+9.1f} "
              f"{per_err:>6.2f}%")
    print(f"{'='*70}")

    return metrics, predictions, targets


def main():
    parser = argparse.ArgumentParser(description='Run Reliability-Aware RUL Prognostics')
    parser.add_argument('--epochs', type=int, default=None, help='Override default epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override default batch size')
    args = parser.parse_args()

    # Update Config if args provided
    if args.epochs: config.EPOCHS = args.epochs
    if args.batch_size: config.BATCH_SIZE = args.batch_size

    # Setup
    config.set_seed()
    config.create_directories()
    device = config.get_device()
    print(f"Using device: {device}\n")

    # 1. Data Loader & Preprocessing
    print("="*70)
    print("1. Data Loading and Preprocessing")
    print("="*70)
    data_loader = DataLoaderFileBase(config)
    data_loader.load_and_split_files()

    # 2. Train STFT Soft Label Generator
    print(f"\n2. Training STFT-based Soft Label Generator...")
    data_loader.train_soft_label_generator()
    output = Path(config.SAVE_DIR)
    source = []
    for split, groups in [('train', data_loader.train_files), ('val', data_loader.val_files),
                          ('test', data_loader.test_files)]:
        for condition, paths in groups.items():
            for path in paths:
                source.append(dict(split=split, condition=condition, path=str(Path(path).resolve()),
                    sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    settings = {k: v for k, v in vars(type(config)).items()
                if k.isupper() and isinstance(v, (str, int, float, bool, list, tuple, dict))}
    with (output/'run_config.json').open('w', encoding='utf-8') as f:
        json.dump(dict(settings=settings, files=source,
                       gate_validation=getattr(data_loader.soft_label_generator, 'validation_protocol', None),
                       protocol='revised legacy BiLSTM path; not an archived paper-table execution'),
                  f, indent=2, default=str)
    generator = data_loader.soft_label_generator
    joblib.dump(generator.scaler, output/'gate_scaler.joblib')
    if hasattr(generator.model, 'state_dict'):
        torch.save(generator.model.state_dict(), output/'gate_state.pt')

    # 3. Train Ensemble Models
    print(f"\n3. Training Condition-Specific Bi-LSTM Ensemble...")
    models, histories, test_data_list = train_ensemble_models(data_loader, config)
    with (output/'training_histories.json').open('w', encoding='utf-8') as f:
        json.dump(histories, f, indent=2)

    # 4. Evaluation
    print(f"\n4. Final Evaluation...")
    metrics, preds, targets = evaluate_framework(models, test_data_list, config)

    # Save Results
    results = {k: float(v) for k, v in metrics.items()}
    with open('results/final_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n✅ Execution Completed Successfully.")
    print("   Results saved to ./results/")


if __name__ == "__main__":
    main()
