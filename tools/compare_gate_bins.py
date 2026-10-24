"""DC versus bins0/1/2 using the same frozen regressors and MC draws."""

from pathlib import Path
import sys, argparse, json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from utils.condition_gate import STFTRateGate, CONDITIONS
from utils.revision_data import read_manifest, sha256
from train_revision import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run = Path(a.run)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    rec, _ = read_manifest(a.manifest, bool(cfg["parent_overlap"]))
    train = [r for r in rec if r["split"] == "train"]
    val = {r["record_id"]: r for r in rec if r["split"] == "val"}
    old = STFTRateGate.load(run / "gate.pkl")
    new = STFTRateGate(bins=(0, 1, 2), seed=old.seed)
    x = []
    y = []
    groups = []
    for r in train:
        for f in [0.7, 0.8, 0.9]:
            z = r["array"][: int(len(r["array"]) * f)]
            x.append(new.features(z[:, 1], z[:, 0]))
            y.append(CONDITIONS.index(r["condition"]))
            groups.append(r["group_id"])
    new.fit(x, y, groups)
    new.save(out / "bins_selected.pkl")
    matched = STFTRateGate(bins=(0, 1, 2), seed=old.seed)
    matched.model = matched.classifier(
        old.selected["kind"], old.selected["C"], old.seed
    ).fit(x, y)
    matched.selected = dict(old.selected)
    matched.save(out / "bins_matched_classifier.pkl")
    save(
        out / "selection_before_val.json",
        dict(
            baseline=old.selected,
            nonzero_selected=new.selected,
            nonzero_cv=new.cv_results,
            matched=matched.selected,
            source="trainonly groupCV ormatched classifier; no valtargetsused",
            normalization="allgates useexposure-normalizedamplitudes; onlyaddedbins1/2",
            regressor_checkpoints_retrained=False,
        ),
    )
    original = pd.read_csv(run / "predictions.csv")
    frames = []
    metrics = []
    for seed in cfg["seeds"]:
        for role in ["experts", "shared"]:
            base = original[
                (original.seed == seed)
                & (original.model == role)
                & (original.gate == "soft")
            ].copy()
            if base.empty:
                raise ValueError("This comparison requires --models all checkpoints.")
            e = base[[f"expert{j}_deterministic" for j in range(3)]].to_numpy()
            draws = (
                np.load(run / f"seed{seed}" / f"mc_components_{role}_soft.npy")
                if cfg["mc_samples"] >= 2
                else None
            )
            for name, gate in [
                ("dc", old),
                ("bins_train_selected", new),
                ("bins_matched_classifier", matched),
            ]:
                w = []
                for row in base.to_dict("records"):
                    z = val[row["record_id"]]["array"][: row["end_index"] + 1]
                    w.append(gate.predict_weights(z[:, 1], z[:, 0]))
                w = np.array(w)
                det = np.maximum(0, (w * e).sum(1))
                d = base[
                    ["record_id", "group_id", "end_index", "observed_cycle", "target"]
                ].copy()
                d["seed"] = seed
                d["model"] = role
                d["gate"] = name
                d["deterministic_prediction"] = det
                d["prediction"] = det
                d["sigma"] = np.nan
                if draws is not None:
                    z = np.maximum(0, (draws * w[None]).sum(-1))
                    d["prediction"] = z.mean(0)
                    d["sigma"] = z.std(0)
                for j in range(3):
                    d[f"weight{j}"] = w[:, j]
                frames.append(d)
                for space, column in [
                    ("deterministic", "deterministic_prediction"),
                    ("MC", "prediction"),
                ]:
                    error = d[column] - d.target
                    metrics.append(
                        dict(
                            seed=seed,
                            model=role,
                            gate=name,
                            prediction_type=space,
                            mae=float(abs(error).mean()),
                            rmse=float(np.sqrt((error**2).mean())),
                        )
                    )
    pd.concat(frames).to_csv(out / "predictions.csv", index=False)
    d = pd.DataFrame(metrics)
    d.to_csv(out / "metrics_by_seed.csv", index=False)
    s = (
        d.groupby(["model", "gate", "prediction_type"])
        .agg(
            mae=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
        )
        .reset_index()
    )
    s.to_csv(out / "summary.csv", index=False)
    for r in rec:
        if sha256(r["path"]) != r["sha256"]:
            raise ValueError("Source input changed.")
    save(
        out / "complete.json",
        dict(passed=True, all_variants_reported=True, source_csv_changed=False),
    )
    print(s.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
