"""Explicit manifest, measured cycle inputs, causal PHI and train-only scaling."""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .condition_gate import CONDITIONS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest(path, allow_overlap=False):
    path = Path(path).resolve()
    d = pd.read_csv(path, dtype=str).fillna("")
    required = {"record_id", "path", "split", "condition", "group_id"}
    if not required.issubset(d):
        raise ValueError(f"Manifest requires {sorted(required)}")
    if d.record_id.duplicated().any() or not set(d.split).issubset(
        {"train", "val", "test"}
    ):
        raise ValueError("Unique record IDs and train/val/test split names required.")
    rows = []
    for r in d.to_dict("records"):
        p = Path(r["path"])
        p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
        a = pd.read_csv(p)
        cols = ["cumulative_contact", "res", "force", "RUL"]
        if not set(cols).issubset(a):
            raise ValueError(
                f"{p}: needs explicit {cols}; RUL is never silently created."
            )
        x = a[cols].to_numpy(dtype=float)
        if len(x) < 400 or not np.isfinite(x).all() or (np.diff(x[:, 0]) <= 0).any():
            raise ValueError(f"{p}: need >=400 finite rows and increasing cycles.")
        if (x[:, 3] < 0).any() or (x[:, 2] <= 0).any():
            raise ValueError(f"{p}: negative RUL or nonpositive force.")
        if r["split"] == "train" and r["condition"] not in CONDITIONS:
            raise ValueError("Only training conditions 0A/0.5A/1.0A train the experts.")
        if not r["group_id"]:
            raise ValueError(
                "Supply original-parent group IDs; do not split augmented siblings."
            )
        r.update(path=str(p), sha256=sha256(p), array=x)
        rows.append(r)
    train = [r for r in rows if r["split"] == "train"]
    if {r["condition"] for r in train} != set(CONDITIONS):
        raise ValueError("Need training files for all three conditions.")
    overlap = sorted(
        {r["group_id"] for r in train}
        & {r["group_id"] for r in rows if r["split"] != "train"}
    )
    if overlap and not allow_overlap:
        raise ValueError(
            "Train/evaluation parent groups overlap. Use --allow-development-overlap only for disclosed development replay."
        )
    return rows, overlap


def input_features(a, alpha=2.34, beta=0.133):
    r, f = a[:, 1], a[:, 2]
    residual = abs(r - (alpha * np.maximum(f, 1e-10) ** (-1 / 3) + beta))
    return np.column_stack(
        [
            np.clip((residual - residual[:100].mean()) / 30.0, 0, 1),
            r,
            f,
            (r - r[:100].mean()) / (r[:100].mean() + 1e-10),
        ]
    ).astype("float32")


def fit_scaler(records, alpha, beta):
    x = np.asfortranarray(
        np.concatenate([input_features(r["array"], alpha, beta) for r in records])
    )
    return dict(mean=x.mean(0), scale=np.maximum(x.std(0), 1e-5))


def windows(
    records,
    scaler,
    phi=True,
    training=True,
    fractions=(0.7, 0.8, 0.9),
    width=200,
    stride=100,
    alpha=2.34,
    beta=0.133,
):
    x, y, labels, meta = [], [], [], []
    for r in records:
        a = r["array"]
        features = (input_features(a, alpha, beta) - scaler["mean"]) / scaler["scale"]
        if not phi:
            features[:, 0] = 0
        ends = (
            range(int(len(a) * 0.6) + width - 1, len(a), stride)
            if training
            else [int(len(a) * f) - 1 for f in fractions]
        )
        for end in ends:
            if end < width - 1:
                raise ValueError("Observation prefix shorter than network window.")
            cyc = np.full(width, a[end, 0] / 380000.0, dtype="float32")
            x.append(np.column_stack([features[end - width + 1 : end + 1], cyc]))
            y.append(a[end, 3])
            labels.append(
                CONDITIONS.index(r["condition"]) if r["condition"] in CONDITIONS else -1
            )
            meta.append(
                dict(
                    record_id=r["record_id"],
                    group_id=r["group_id"],
                    folder_condition=r["condition"],
                    end_index=end,
                    observed_cycle=a[end, 0],
                    target=a[end, 3],
                    observed_rows=end + 1,
                    total_rows=len(a),
                )
            )
    return (
        np.array(x, dtype="float32"),
        np.array(y, dtype="float32"),
        np.array(labels),
        meta,
    )
