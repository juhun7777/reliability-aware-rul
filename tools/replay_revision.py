"""Replay revised checkpoints, fixed MC seeds, inputs and all exported metrics."""

from pathlib import Path
import sys, argparse, json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import numpy as np, pandas as pd, torch
from train_revision import evaluate, save
from utils.condition_gate import STFTRateGate, CONDITIONS
from utils.revision_data import read_manifest, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()
    run = Path(a.run).resolve()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    records, overlap = read_manifest(
        a.manifest, allow_overlap=bool(config["parent_overlap"])
    )
    expected = {r["record_id"]: r for r in config["source"]}
    if set(expected) != {r["record_id"] for r in records}:
        raise ValueError("Replay input IDs differ.")
    for r in records:
        for k in ["sha256", "condition", "group_id", "split"]:
            if expected[r["record_id"]][k] != r[k]:
                raise ValueError(f'Replay input {r["record_id"]}: {k} differs.')
    args = SimpleNamespace(
        **{
            k: v
            for k, v in config.items()
            if k
            not in [
                "source",
                "parent_overlap",
                "manifest_sha256",
                "scope",
                "inputs",
                "target_scale",
                "gate_classes",
                "uncertainty",
            ]
        }
    )
    if a.deterministic:
        args.mc_samples = 0
    gate = STFTRateGate.load(run / "gate.pkl")
    evaluation = [r for r in records if r["split"] == "val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = []
    metrics = []
    for seed in config["seeds"]:
        args.active_seed = seed
        paths = {}
        folder = run / f"seed{seed}"
        dest = out / f"seed{seed}"
        dest.mkdir()
        for role in (
            ["pooled", "experts", "shared"]
            if config["models"] == "all"
            else [config["models"]]
        ):
            paths[role] = (
                [folder / (role + "_" + c.replace(".", "p")) for c in CONDITIONS]
                if role == "experts"
                else [folder / role]
            )
            for d in paths[role]:
                info = json.loads((d / "config.json").read_text(encoding="utf-8"))
                if sha256(d / "model.pt") != info["checkpoint_sha256"]:
                    raise ValueError("Checkpoint hash differs.")
        d, m = evaluate(paths, evaluation, gate, args, device, dest)
        pred.append(d)
        metrics.extend(m)
    result = pd.concat(pred, ignore_index=True)
    result.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics_by_seed.csv", index=False)
    original = pd.read_csv(run / "predictions.csv")
    keys = ["seed", "model", "gate", "record_id", "end_index"]
    joint = result.merge(original, on=keys, suffixes=("_replay", "_original"))
    if len(joint) != len(original):
        raise ValueError("Prediction row identities differ.")
    column = "deterministic_prediction" if a.deterministic else "prediction"
    maximum = float(abs(joint[column + "_replay"] - joint[column + "_original"]).max())
    # float32 predictions use a shortdecimal CSV representation; tolerance is
    # less than0.02cycle versus100cycle observations, not an accuracy tolerance.
    if maximum >= 0.02:
        raise ValueError(f"Replay mismatch {maximum} cycles.")
    sigma_delta = None
    if not a.deterministic and config["mc_samples"] >= 2:
        sigma_delta = float(abs(joint.sigma_replay - joint.sigma_original).max())
        if sigma_delta >= 0.02:
            raise ValueError("Sigma replay mismatch.")
    save(
        out / "verification.json",
        dict(
            passed=True,
            rows=len(joint),
            input_files=len(records),
            prediction_max_difference_cycles=maximum,
            sigma_max_difference_cycles=sigma_delta,
            mc_seed_rule="seed+100000",
            deterministic=a.deterministic,
        ),
    )
    print("REPLAY PASSED", maximum, sigma_delta, flush=True)


if __name__ == "__main__":
    main()
