"""Condition similarity from observed STFT amplitudes and elapsed exposure.

The DC-only representation is also a Hann-weighted moving mean. It must not
be presented as evidence that nonzero frequency bins improve predictions.
"""

import pickle
from pathlib import Path
import numpy as np
from scipy.signal import stft
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CONDITIONS = ("0A", "0.5A", "1.0A")


class STFTRateGate:
    def __init__(self, bins=(0,), exposure_normalization=True, seed=42):
        self.bins = tuple(bins)
        self.exposure_normalization = exposure_normalization
        self.seed = seed
        self.model = None

    def features(self, resistance, cycles):
        r = np.asarray(resistance, dtype=float)
        c = np.asarray(cycles, dtype=float)
        if r.ndim != 1 or c.shape != r.shape or len(r) < 256:
            raise ValueError("At least 256 observed R/cycle rows are required.")
        if not np.isfinite(r).all() or not np.isfinite(c).all():
            raise ValueError("Nonfinite gate input.")
        step = np.diff(c)
        if (step <= 0).any() or not np.allclose(step, step[0]) or c[-1] <= 0:
            raise ValueError("STFT requires an increasing, uniform cycle grid.")
        baseline = r[:100].mean()
        _, time, z = stft(
            r - baseline, fs=1.0, nperseg=256, noverlap=128, boundary=None, padded=False
        )
        time = time * step[0] / 100000.0
        time = time - time.mean()
        if time @ time == 0:
            raise ValueError("Need multiple complete STFT frames; use >=384 rows.")
        amp = abs(z[np.asarray(self.bins)])
        summary = np.column_stack(
            [amp.mean(-1), amp.std(-1), *np.percentile(amp, [25, 50, 75], axis=-1)]
        )
        if self.exposure_normalization:
            summary = summary / (c[-1] / 100000.0)
        slope = amp @ time / (time @ time)
        return np.concatenate([summary.ravel(), slope, [baseline, c[-1] / 380000.0]])

    @staticmethod
    def classifier(kind, c, seed):
        if kind == "logistic":
            m = LogisticRegression(
                C=c, solver="lbfgs", max_iter=1500, class_weight="balanced"
            )
        else:
            m = SVC(C=c, probability=True, class_weight="balanced", random_state=seed)
        return make_pipeline(StandardScaler(), m)

    def fit(self, x, labels, groups, select=True):
        x = np.asarray(x, dtype=float)
        y = np.asarray(labels, dtype=int)
        groups = np.asarray(groups)
        if set(y) != {0, 1, 2}:
            raise ValueError("Training gate needs all three source classes.")
        candidates = [("svm", 1.0)]
        self.cv_results = []
        if select:
            n = min(5, *(len(set(groups[y == j])) for j in range(3)))
            if n < 2:
                raise ValueError(
                    "Need >=2 original groups per class for gate selection."
                )
            folds = list(
                StratifiedGroupKFold(
                    n_splits=n, shuffle=True, random_state=self.seed
                ).split(x, y, groups)
            )
            for a, b in folds:
                if set(y[a]) != {0, 1, 2} or set(y[b]) != {0, 1, 2}:
                    raise ValueError("Group folds must contain all three classes.")
                assert not (set(groups[a]) & set(groups[b]))
            candidates = [("logistic", c) for c in [0.01, 0.1, 1.0, 10.0]] + [
                ("svm", c) for c in [0.1, 1.0, 10.0]
            ]
            for kind, c in candidates:
                p = np.zeros((len(y), 3))
                for a, b in folds:
                    m = self.classifier(kind, c, self.seed).fit(x[a], y[a])
                    p[b] = m.predict_proba(x[b])
                self.cv_results.append(
                    dict(
                        kind=kind,
                        C=c,
                        logloss=float(log_loss(y, p, labels=[0, 1, 2])),
                        accuracy=float(accuracy_score(y, p.argmax(1))),
                    )
                )
            best = min(self.cv_results, key=lambda v: (v["logloss"], v["kind"], v["C"]))
            candidates = [(best["kind"], best["C"])]
        self.selected = dict(
            kind=candidates[0][0],
            C=candidates[0][1],
            selection="train_group_cv" if select else "fixed_training_config",
        )
        self.model = self.classifier(*candidates[0], self.seed).fit(x, y)
        return self

    def predict_weights(self, resistance, cycles, power=1.0):
        if self.model is None:
            raise RuntimeError("Fit/load the gate before inference.")
        if not np.isfinite(power) or power <= 0:
            raise ValueError("power must be positive.")
        p = self.model.predict_proba(self.features(resistance, cycles)[None])[0]
        w = np.maximum(p, 1e-12) ** power
        return w / w.sum()

    def save(self, path):
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with Path(path).open("rb") as f:
            return pickle.load(f)  # load only trusted model artifacts
