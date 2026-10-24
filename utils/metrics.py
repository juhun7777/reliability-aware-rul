"""
Metrics Module
IEEE Paper: Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins

Evaluation Metrics (Section IV-D, Equations 12-15)
"""
import numpy as np


def calculate_rmse(predictions, targets):
    """
    Root Mean Squared Error (Equation 12)
    RMSE = sqrt(1/N * sum((y_hat - y)^2))
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))


def calculate_mae(predictions, targets):
    """
    Mean Absolute Error (Equation 13)
    MAE = 1/N * sum(|y_hat - y|)
    """
    return np.mean(np.abs(predictions - targets))


def calculate_mape(predictions, targets):
    """
    Mean Absolute Percentage Error (Equation 14)
    MAPE = 100%/N * sum(|(y - y_hat) / y|)
    """
    return np.mean(np.abs((targets - predictions) / (targets + 1e-10))) * 100


def calculate_nasa_score(predictions, targets):
    """
    NASA Prognostic Score (Equation 15)
    Paper: a1 = 13, a2 = 10

    S = sum(exp(-d_i/a1) - 1) if d_i < 0 (Early prediction)
      + sum(exp(d_i/a2) - 1)  if d_i >= 0 (Late prediction)

    where d_i = y_hat_i - y_i

    Returns log10(1 + sum(scores)) after scaling errors by 10,000 cycles.
    This is a scaled/log diagnostic, not the unscaled NASA challenge total.
    """
    errors = predictions - targets

    # Scale errors to prevent overflow (10K cycles = 1 unit)
    scale = 10000.0
    errors_scaled = errors / scale

    scores = []
    MAX_EXP = 30

    for e in errors_scaled:
        if e < 0:
            # Early prediction (a1 = 13).
            exponent = min(-e / 13, MAX_EXP)
            score = np.exp(exponent) - 1
        else:
            # Late prediction has the stronger penalty (a2 = 10).
            exponent = min(e / 10, MAX_EXP)
            score = np.exp(exponent) - 1

        scores.append(score)

    total_score = np.sum(scores)

    if not np.isfinite(total_score):
        raise ValueError('Nonfinite NASA score; inspect predictions and targets.')

    log_score = np.log10(total_score + 1)
    return log_score


def calculate_r2_score(predictions, targets):
    """
    R-squared (Coefficient of Determination)
    R2 = 1 - SS_res / SS_tot
    """
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-10))


def calculate_max_error(predictions, targets):
    """Maximum Absolute Error"""
    return np.max(np.abs(predictions - targets))


def calculate_median_ae(predictions, targets):
    """Median Absolute Error"""
    return np.median(np.abs(predictions - targets))


def calculate_std_error(predictions, targets):
    """Standard Deviation of Errors"""
    return np.std(predictions - targets)


def evaluate_predictions(predictions, targets, predictions_orig=None, targets_orig=None):
    """
    Comprehensive evaluation

    Args:
        predictions: Normalized predictions
        targets: Normalized targets
        predictions_orig: Original scale predictions (optional)
        targets_orig: Original scale targets (optional)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    metrics = {
        'normalized': {
            'rmse': calculate_rmse(predictions, targets),
            'mae': calculate_mae(predictions, targets),
            'r2': calculate_r2_score(predictions, targets)
        }
    }

    if predictions_orig is not None and targets_orig is not None:
        metrics['original'] = {
            'rmse': calculate_rmse(predictions_orig, targets_orig),
            'mae': calculate_mae(predictions_orig, targets_orig),
            'mape': calculate_mape(predictions_orig, targets_orig),
            'nasa_score': calculate_nasa_score(predictions_orig, targets_orig),
            'r2': calculate_r2_score(predictions_orig, targets_orig),
            'max_error': calculate_max_error(predictions_orig, targets_orig),
            'median_ae': calculate_median_ae(predictions_orig, targets_orig),
            'std_error': calculate_std_error(predictions_orig, targets_orig)
        }

    return metrics


def print_metrics(metrics, split_name='Test'):
    """Print evaluation metrics"""
    print("="*70)
    print(f"[{split_name}] Results (Normalized [0,1]):")
    print(f"  RMSE: {metrics['normalized']['rmse']:.4f}")
    print(f"  MAE: {metrics['normalized']['mae']:.4f}")
    print(f"  R2: {metrics['normalized']['r2']:.4f}")

    if 'original' in metrics:
        print(f"\n[{split_name}] Results (Original Cycles):")
        print(f"  RMSE:       {metrics['original']['rmse']:>10.2f} cycles")
        print(f"  MAE:        {metrics['original']['mae']:>10.2f} cycles")
        print(f"  MAPE:       {metrics['original']['mape']:>10.2f}%")
        print(f"  NASA Score: {metrics['original']['nasa_score']:>10.2f}")
        print(f"  R2:         {metrics['original']['r2']:>10.4f}")

        if 'max_error' in metrics['original']:
            print(f"\n  Additional Metrics:")
            print(f"  Max Error:  {metrics['original']['max_error']:>10.2f} cycles")
            print(f"  Median AE:  {metrics['original']['median_ae']:>10.2f} cycles")
            print(f"  Std Error:  {metrics['original']['std_error']:>10.2f} cycles")

    print("="*70)
