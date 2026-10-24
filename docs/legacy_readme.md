# Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins

<!-- Badges -->
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Submission Status](https://img.shields.io/badge/IEEE-Trans.%20Reliability-green)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=24)

> **Official implementation code for the IEEE Transactions on Reliability submission.**

## Paper Information
**Title:** Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins Using Physics-Informed Residuals
**Authors:** Juhun Lee, Kihyun Kim, Hyo-Young Kim*
**Affiliation:** Tech University of Korea, Department of Mechatronics Engineering
**Journal:** IEEE Transactions on Reliability (Under Review)

---

## Overview

This repository contains the source code for a novel **Physics-Informed Uncertainty-Aware Ensemble Framework** designed to predict the Remaining Useful Life (RUL) of semiconductor test socket micro-pins. The proposed method addresses the challenge of predicting contact failure in varying current conditions (0A, 0.5A, 1.0A) by integrating physical contact theory with deep learning.

### Key Contributions
1.  **Physics-Informed Health Indicator (PHI)**: Utilizes the **Holm-Hertz contact theory** to define a degradation indicator, separating physical contact resistance changes from degradation.
    *   Formula: $R_{ideal}(F) = \alpha \cdot F^{-1/3} + \beta$ (where $\alpha=2.34, \beta=0.133$)
2.  **STFT-based Soft Labeling**: Proposes a soft-labeling technique using Short-Time Fourier Transform (STFT) features to interpolate operating conditions for robust generalization.
3.  **Condition-Specific Ensemble**: Deploys a weighted ensemble of Bi-LSTM models with **Bahdanau Attention**, specialized for different current levels.
4.  **Uncertainty Quantification**: Implements **Monte Carlo Dropout** to provide confidence intervals for RUL predictions, critical for industrial reliability.

---

## Model Architecture

The overall framework consists of three main stages: **Preprocessing (PHI & Soft Labeling)**, **Ensemble Modeling**, and **RUL Prediction**.

![Overall Framework](assets/framework.jpg)

---

## Project Structure

```bash
gitcode/
├── config/
│   └── config.py           # Hyperparameters (Table IV from paper)
├── models/
│   ├── lstm.py             # Bi-LSTM with Bahdanau Attention (Prop. Method)
│   └── ensemble_lstm.py    # Weighted Ensemble Logic
├── utils/
│   ├── preprocessing.py    # PHI Calculation(Algo 1) & STFT Soft Labeling(Algo 2)
│   ├── data_loader.py      # Data Pipeline & Splitting
│   └── metrics.py          # Evaluation Metrics (RMSE, NASA Score)
├── train.py                # Main Entry Point for Training & Evaluation
├── requirements.txt        # Python Dependencies
└── README.md               # Project Documentation
```

---

## Hyperparameters (Paper Table IV)

| Category | Parameter | Value |
|:---|:---|:---|
| **Optimization** | Optimizer | AdamW |
| | Learning Rate | 5×10⁻⁴ |
| | Batch Size | 32 |
| | Loss Function | Huber Loss (δ=0.5) |
| **Model** | LSTM Layers | 3 (Bidirectional) |
| | Hidden Size | 128 |
| | Attention | Bahdanau-style |
| **PHI** | $\alpha$ (Roughness Coeff) | 2.34 |
| | $\beta$ (Const. Resistance) | 0.133 |

---

## Getting Started

### 1. Prerequisites
*   Python 3.8+
*   PyTorch 1.10+
*   See `requirements.txt` for full list.

### 2. Installation
```bash
git clone https://github.com/juhun7777/reliability-aware-rul.git
cd reliability-aware-rul
pip install -r requirements.txt
```

### 3. Data Structure
The repository includes sample data for code verification. The data is organized as follows:

```
data/
├── train/          # Training data (Known conditions)
│   ├── 0A/         # 0A current condition
│   ├── 0.5A/       # 0.5A current condition
│   └── 1.0A/       # 1.0A current condition
└── test/           # Test data (Unseen conditions for generalization)
    ├── 0.3A/       # Unseen condition
    ├── 0.7A/       # Unseen condition
    └── 0.9A/       # Unseen condition
```

> **Note**: Sample data is provided for code verification. Full dataset will be made available upon paper acceptance.

### 4. Training & Evaluation
Run the main script to reproduce the results. This script handles data loading, soft-label generator training, ensemble model training, and final evaluation.
```bash
python train.py
```
> **Note**: The script automatically performs **Last Window Evaluation** and **Uncertainty Quantification** (MC Dropout) as described in the paper.

---

## Experimental Results (Paper Table V)

The proposed method demonstrates superior performance compared to state-of-the-art baselines.

| Model | RMSE (cycles) | MAE (cycles) | NASA Score |
|:---|:---:|:---:|:---:|
| **Proposed Framework** | **24,246** | **17,809** | **0.42** |
| Autoformer | 23,215 | 20,617 | 0.45 |
| FEDformer | 24,649 | 22,430 | 0.47 |

---

## Baseline Models

For fair comparison, we evaluated against the following state-of-the-art baselines. We used official implementations with hyperparameters tuned for our RUL prediction task:

| Model | Type | Reference |
|:------|:-----|:----------|
| **XGBoost** | Machine Learning | [Official Library](https://xgboost.readthedocs.io/) |
| **LightGBM** | Machine Learning | [Official Library](https://lightgbm.readthedocs.io/) |
| **CatBoost** | Machine Learning | [Official Library](https://catboost.ai/) |
| **Autoformer** | Transformer | [Wu et al., NeurIPS 2021](https://github.com/thuml/Autoformer) |
| **FEDformer** | Transformer | [Zhou et al., ICML 2022](https://github.com/MAZiqing/FEDformer) |

> **Note**: Hyperparameters for baseline models are provided in `config/config.py` for reproducibility.

---

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{lee2026reliability,
  title={Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins Using Physics-Informed Residuals},
  author={Lee, Juhun and Kim, Kihyun and Kim, Hyo-Young},
  journal={IEEE Transactions on Reliability},
  year={2026},
  publisher={IEEE}
}
```

---

## Contact
For any queries, please contact:
*   **Hyo-Young Kim** (Corresponding Author): kimhy@tukorea.ac.kr
*   **Juhun Lee**: 2018144024@tukorea.ac.kr
