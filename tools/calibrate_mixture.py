"""Select pooled/expert mixing from training-internal groups, never val labels."""

from pathlib import Path
import sys, argparse, json
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, torch
from models.condition_transformer import ConditionTransformer
from utils.condition_gate import STFTRateGate, CONDITIONS
from utils.revision_data import read_manifest, windows, sha256
from train_revision import predict, save


def paths(folder, role):
    return (
        [folder / ("experts_" + c.replace(".", "p")) for c in CONDITIONS]
        if role == "experts"
        else [folder / role]
    )


def infer_role(folder, role, records, args, device, inner):
    outputs = []
    metadata = None
    target = None
    for d in paths(folder, role):
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        sc = dict(np.load(d / ("inner_scaler.npz" if inner else "scaler.npz")))
        x, y, _, meta = windows(
            records,
            sc,
            not args.no_phi,
            False,
            tuple(args.fractions),
            alpha=args.alpha,
            beta=args.beta,
        )
        m = ConditionTransformer(heads=config["heads"]).to(device)
        m.load_state_dict(
            torch.load(
                d / ("inner.pt" if inner else "model.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        p, _ = predict(m, x, device, 0)
        outputs.append(p)
        metadata = meta
        target = y
    return np.column_stack(outputs), target, metadata


def weights(gate, records, meta):
    lookup = {r["record_id"]: r for r in records}
    out = []
    for o in meta:
        a = lookup[o["record_id"]]["array"][: o["end_index"] + 1]
        out.append(gate.predict_weights(a[:, 1], a[:, 0]))
    return np.array(out)


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
    args = SimpleNamespace(**cfg)
    if cfg["models"] != "all":
        raise ValueError("Requires --models all checkpoints.")
    records, _ = read_manifest(a.manifest, allow_overlap=bool(cfg["parent_overlap"]))
    train = [r for r in records if r["split"] == "train"]
    gate_full = STFTRateGate.load(run / "gate.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    selected = []
    all_cal = []
    # Stage1 never accesses val observations, predictions or targets.
    for seed in cfg["seeds"]:
        folder = run / f"seed{seed}"
        pcfg = json.loads(
            (folder / "pooled" / "config.json").read_text(encoding="utf-8")
        )
        ids = set(pcfg["inner_calibration_ids"])
        cal = [r for r in train if r["record_id"] in ids]
        inside = [r for r in train if r["record_id"] not in ids]
        assert not ({r["group_id"] for r in inside} & {r["group_id"] for r in cal})
        gate = STFTRateGate(
            bins=gate_full.bins,
            exposure_normalization=gate_full.exposure_normalization,
            seed=gate_full.seed,
        )
        gx = []
        gy = []
        for r in inside:
            for f in [0.7, 0.8, 0.9]:
                z = r["array"][: int(len(r["array"]) * f)]
                gx.append(gate.features(z[:, 1], z[:, 0]))
                gy.append(CONDITIONS.index(r["condition"]))
        gate.selected = dict(gate_full.selected)
        gate.model = gate.classifier(
            gate.selected["kind"], gate.selected["C"], gate.seed
        ).fit(gx, gy)
        gate.save(out / f"inner_gate_seed{seed}.pkl")
        pool, y, meta = infer_role(folder, "pooled", cal, args, device, True)
        w = weights(gate, cal, meta)
        for role in ["experts", "shared"]:
            e, _, _ = infer_role(folder, role, cal, args, device, True)
            expert = (e * w).sum(1)
            scores = []
            for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
                pred = np.maximum(0, fraction * pool[:, 0] + (1 - fraction) * expert)
                d = pd.DataFrame(meta)
                d["absolute_error"] = abs(pred - y)
                # Equal original-group weights; repeated augmented observations
                # are not counted as independent calibration specimens.
                score = float(d.groupby("group_id").absolute_error.mean().mean())
                scores.append(
                    dict(
                        seed=seed,
                        role=role,
                        pooled_fraction=fraction,
                        calibration_group_mae=score,
                    )
                )
            best = min(
                scores,
                key=lambda r: (
                    r["calibration_group_mae"],
                    abs(r["pooled_fraction"] - 0.5),
                ),
            )
            selected.append(
                dict(
                    **best,
                    calibration_record_ids=sorted(ids),
                    calibration_group_ids=sorted({r["group_id"] for r in cal}),
                )
            )
            all_cal += scores
    save(
        out / "selection_train_only.json",
        dict(
            selected=selected,
            grid=[0.0, 0.25, 0.5, 0.75, 1.0],
            criterion="equalgroupMAE ontraining-internal calibration; bestepoch andmixing reusecalibration, smallsample development",
            gate="samekind/C asfulltrain-selectedgate, coefficients refit innertraining only",
            val_used_for_selection=False,
        ),
    )
    pd.DataFrame(all_cal).to_csv(out / "calibration_grid.csv", index=False)
    # Stage2 applies frozen choices to unchanged val, reporting allchoices/results.
    val = [r for r in records if r["split"] == "val"]
    frames = []
    metrics = []
    for seed in cfg["seeds"]:
        folder = run / f"seed{seed}"
        pool, y, meta = infer_role(folder, "pooled", val, args, device, False)
        w = weights(gate_full, val, meta)
        for choice in [s for s in selected if s["seed"] == seed]:
            role = choice["role"]
            fraction = choice["pooled_fraction"]
            e, _, _ = infer_role(folder, role, val, args, device, False)
            estimate = np.maximum(
                0, fraction * pool[:, 0] + (1 - fraction) * (e * w).sum(1)
            )
            d = pd.DataFrame(meta)
            d["seed"] = seed
            d["role"] = role
            d["pooled_fraction"] = fraction
            d["prediction"] = estimate
            d["absolute_error"] = abs(estimate - y)
            frames.append(d)
            metrics.append(
                dict(
                    seed=seed,
                    role=role,
                    pooled_fraction=fraction,
                    mae=float(abs(estimate - y).mean()),
                    rmse=float(np.sqrt(np.mean((estimate - y) ** 2))),
                )
            )
    pd.concat(frames).to_csv(out / "predictions.csv", index=False)
    md = pd.DataFrame(metrics)
    md.to_csv(out / "metrics_by_seed.csv", index=False)
    summary = (
        md.groupby("role")
        .agg(
            mae=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out / "summary.csv", index=False)
    for r in records:
        if sha256(r["path"]) != r["sha256"]:
            raise ValueError("Source input changed.")
    save(
        out / "complete.json",
        dict(
            passed=True, selected_before_val_evaluation=True, source_csv_modified=False
        ),
    )
    print(md.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
