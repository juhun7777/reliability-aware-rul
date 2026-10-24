"""Revised, explicit-data training. Historical train.py is a separate protocol."""

from pathlib import Path
import argparse, json, random, time
import numpy as np, pandas as pd, torch
from torch import nn
from models.condition_transformer import ConditionTransformer
from utils.condition_gate import STFTRateGate, CONDITIONS
from utils.revision_data import read_manifest, fit_scaler, windows, sha256
from utils.metrics import calculate_rmse, calculate_mae, calculate_nasa_score


def save(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def predict(model, x, device, draws=0):
    states = [(m, m.training) for m in model.modules()]
    model.eval()
    x = torch.tensor(x, device=device)
    try:
        with torch.inference_mode():
            deterministic = (
                torch.cat([model(b) for b in x.split(128)]).cpu().numpy() * 100000.0
            )
            if draws < 2:
                return deterministic, None
            # Activate explicit dropout modules, keeping normalisation and other
            # layers in evaluation mode. Gate weights are fixed across MC draws.
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
            mc = np.stack(
                [
                    torch.cat([model(b) for b in x.split(128)]).cpu().numpy() * 100000.0
                    for _ in range(draws)
                ]
            )
            return deterministic, mc
    finally:
        for m, training in states:
            m.training = training


def fit(model, train, calibration, sc, args, device, path, epochs=None, schedule=None):
    seed_all(args.active_seed)
    x, y, c, _ = windows(train, sc, not args.no_phi, alpha=args.alpha, beta=args.beta)
    x = torch.tensor(x, device=device)
    y = torch.tensor(y / 100000.0, device=device)
    c = torch.tensor(c, device=device)
    if calibration:
        v, yv, cv, _ = windows(
            calibration,
            sc,
            not args.no_phi,
            training=False,
            alpha=args.alpha,
            beta=args.beta,
        )
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)
    best = float("inf")
    best_epoch = 0
    stall = 0
    history = []
    start = time.time()
    for ep in range(args.epochs if epochs is None else epochs):
        if schedule is not None:
            for group in opt.param_groups:
                group["lr"] = schedule[ep]
        lr = opt.param_groups[0]["lr"]
        model.train()
        losses = []
        for ix in torch.randperm(len(x), device=device).split(args.batch_size):
            opt.zero_grad(set_to_none=True)
            p = model(x[ix])
            p = (
                p[torch.arange(len(ix), device=device), c[ix]]
                if p.shape[1] == 3
                else p[:, 0]
            )
            loss = nn.functional.huber_loss(p, y[ix], delta=0.5)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            losses.append(float(loss.detach()))
        row = dict(epoch=ep + 1, train_loss=float(np.mean(losses)), lr=lr)
        if calibration:
            pred, _ = predict(model, v, device)
            p = pred[np.arange(len(pred)), cv] if pred.shape[1] == 3 else pred[:, 0]
            score = float(calculate_rmse(p, yv))
            scheduler.step(score)
            row["calibration_rmse"] = score
            if score < best - 1.0:
                best = score
                best_epoch = ep + 1
                stall = 0
                torch.save(
                    {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    path,
                )
            else:
                stall += 1
        history.append(row)
        if calibration and stall >= args.patience:
            break
    if not calibration:
        best_epoch = epochs
        torch.save(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path
        )
    save(path.with_suffix(".history.json"), history)
    return (
        dict(
            selected_epoch=best_epoch,
            train_files=len(train),
            train_windows=len(x),
            parameters=sum(p.numel() for p in model.parameters()),
            seconds=time.time() - start,
        ),
        history,
    )


def internal_split(train, seed):
    selected = set()
    for condition in CONDITIONS:
        groups = sorted({r["group_id"] for r in train if r["condition"] == condition})
        if len(groups) < 2:
            raise ValueError("Need >=2 original groups per training condition.")
        selected.add(groups[seed % len(groups)])
    a = [r for r in train if r["group_id"] not in selected]
    b = [r for r in train if r["group_id"] in selected]
    if {r["condition"] for r in a} != set(CONDITIONS):
        raise ValueError("Internal groups leave a training class empty.")
    return a, b


def train_models(train, args, device, out):
    inner, cal = internal_split(train, args.active_seed)
    sc = fit_scaler(train, args.alpha, args.beta)
    sci = fit_scaler(inner, args.alpha, args.beta)
    roles = ["pooled", "experts", "shared"] if args.models == "all" else [args.models]
    paths = {}
    for role in roles:
        model_paths = []
        for condition in CONDITIONS if role == "experts" else [None]:
            name = (
                role if condition is None else role + "_" + condition.replace(".", "p")
            )
            folder = out / name
            folder.mkdir()
            t = train
            i = inner
            v = cal
            if condition:
                t = [r for r in train if r["condition"] == condition]
                i = [r for r in inner if r["condition"] == condition]
                v = [r for r in cal if r["condition"] == condition]
            seed_all(args.active_seed)
            heads = 3 if role == "shared" else 1
            info, hist = fit(
                ConditionTransformer(heads=heads),
                i,
                v,
                sci,
                args,
                device,
                folder / "inner.pt",
            )
            epochs = info["selected_epoch"]
            seed_all(args.active_seed)
            full, _ = fit(
                ConditionTransformer(heads=heads),
                t,
                [],
                sc,
                args,
                device,
                folder / "model.pt",
                epochs,
                [h["lr"] for h in hist[:epochs]],
            )
            np.savez(folder / "scaler.npz", **sc)
            np.savez(folder / "inner_scaler.npz", **sci)
            save(
                folder / "config.json",
                dict(
                    role=role,
                    condition=condition,
                    heads=heads,
                    seed=args.active_seed,
                    selected_epoch=epochs,
                    inner_train_ids=[r["record_id"] for r in i],
                    inner_calibration_ids=[r["record_id"] for r in v],
                    full_train_ids=[r["record_id"] for r in t],
                    full=full,
                    checkpoint_sha256=sha256(folder / "model.pt"),
                ),
            )
            model_paths.append(folder)
            print(f"{args.active_seed} {name} selected epochs={epochs}", flush=True)
        paths[role] = model_paths
    return paths


def evaluate(paths, evaluation, gate, args, device, out):
    # MC reproducibility must not depend on how many optimization steps ran.
    seed_all(args.active_seed + 100000)
    sc = dict(np.load(next(iter(paths.values()))[0] / "scaler.npz"))
    x, y, _, meta = windows(
        evaluation,
        sc,
        not args.no_phi,
        False,
        tuple(args.fractions),
        alpha=args.alpha,
        beta=args.beta,
    )
    byid = {r["record_id"]: r for r in evaluation}
    w = []
    for o in meta:
        a = byid[o["record_id"]]["array"][: o["end_index"] + 1]
        w.append(gate.predict_weights(a[:, 1], a[:, 0], power=1.0))
    w = np.array(w)
    outputs = {}
    samples = {}
    frames = []
    for role, folders in paths.items():
        es = []
        mc = []
        for folder in folders:
            cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            m = ConditionTransformer(heads=cfg["heads"]).to(device)
            m.load_state_dict(
                torch.load(folder / "model.pt", map_location=device, weights_only=True)
            )
            p, draw = predict(m, x, device, args.mc_samples)
            es.append(p)
            if draw is not None:
                mc.append(draw)
        outputs[role] = np.column_stack(es)
        if mc:
            samples[role] = np.concatenate(mc, axis=-1)

    def add(role, kind, e, weight, draws=None):
        deterministic = np.maximum(0, (e * weight).sum(1))
        d = pd.DataFrame(meta)
        d["seed"] = args.active_seed
        d["model"] = role
        d["gate"] = kind
        d["deterministic_prediction"] = deterministic
        d["prediction"] = deterministic
        d["sigma"] = np.nan
        if draws is not None:
            raw = (draws * weight[None]).sum(-1)
            z = np.maximum(0, raw)
            d["prediction"] = z.mean(0)
            d["sigma"] = z.std(0)
            d["mc_mean_before_clipping"] = raw.mean(0)
            d["mc_clipped_fraction"] = (raw < 0).mean(0)
            np.save(out / f"mc_{role}_{kind}.npy", z)
            np.save(out / f"mc_components_{role}_{kind}.npy", draws)
        d["error"] = d.prediction - d.target
        for j in range(e.shape[1]):
            d[f"expert{j}"] = draws[:, :, j].mean(0) if draws is not None else e[:, j]
            d[f"expert{j}_deterministic"] = e[:, j]
            d[f"weight{j}"] = weight[:, j]
        for j in range(3):
            d[f"gate_probability{j}"] = w[:, j]
        frames.append(d)

    for role, e in outputs.items():
        if role == "pooled":
            add(role, "none", e, np.ones((len(x), 1)), samples.get(role))
            continue
        for name, weight in [
            ("soft", w),
            ("hard", np.eye(3)[w.argmax(1)]),
            ("uniform", np.full_like(w, 1 / 3)),
        ]:
            add(role, name, e, weight, samples.get(role))
        if "pooled" in outputs:
            e4 = np.column_stack([outputs["pooled"], e])
            w4 = np.column_stack([np.full(len(x), 0.5), w * 0.5])
            draw = (
                np.concatenate([samples["pooled"], samples[role]], axis=-1)
                if role in samples and "pooled" in samples
                else None
            )
            add(role, "pooled50_soft50", e4, w4, draw)
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    metrics = []
    for (model, kind), z in d.groupby(["model", "gate"]):
        metrics.append(
            dict(
                seed=args.active_seed,
                model=model,
                gate=kind,
                n=len(z),
                mae=float(calculate_mae(z.prediction.to_numpy(), z.target.to_numpy())),
                rmse=float(
                    calculate_rmse(z.prediction.to_numpy(), z.target.to_numpy())
                ),
                scaled_log_nasa=float(
                    calculate_nasa_score(z.prediction.to_numpy(), z.target.to_numpy())
                ),
            )
        )
    return d, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--models", choices=["pooled", "experts", "shared", "all"], default="all"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    p.add_argument("--alpha", type=float, default=2.34)
    p.add_argument("--beta", type=float, default=0.133)
    p.add_argument("--no-phi", action="store_true")
    p.add_argument("--mc-samples", type=int, default=0)
    p.add_argument("--gate-bins", choices=["dc", "nonzero_control"], default="dc")
    p.add_argument("--fixed-gate", action="store_true")
    p.add_argument("--allow-development-overlap", action="store_true")
    a = p.parse_args()
    if a.epochs < 1 or a.patience < 1 or any(not 0 < f < 1 for f in a.fractions):
        p.error(
            "Positive epochs/patience and fractions strictly between0and1 required."
        )
    data, overlap = read_manifest(a.manifest, a.allow_development_overlap)
    train = [r for r in data if r["split"] == "train"]
    evaluation = [r for r in data if r["split"] == "val"]
    if not evaluation:
        p.error("Manifest needs fixed val observations for this development protocol.")
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(
        out / "config.json",
        dict(
            **vars(a),
            manifest_sha256=sha256(a.manifest),
            parent_overlap=overlap,
            scope="development comparison; val is not used for epoch/gate selection",
            source=[{k: v for k, v in r.items() if k != "array"} for r in data],
            inputs=["PHI", "res", "force", "relative_res", "observed_cycle/380000"],
            target_scale=100000.0,
            gate_classes=list(CONDITIONS),
            mc_seed_rule="seed+100000, independent of optimization RNG consumption",
            uncertainty="explicit dropout modules; fixed gate; clip each weighted MC draw>=0, then mean/std; component MC means and deterministic outputs are separate; not calibrated coverage",
        ),
    )
    gate = STFTRateGate(
        bins=(0,) if a.gate_bins == "dc" else (0, 1, 2), seed=a.seeds[0]
    )
    gx = []
    gy = []
    groups = []
    for r in train:
        for f in [0.7, 0.8, 0.9]:
            z = r["array"][: int(len(r["array"]) * f)]
            gx.append(gate.features(z[:, 1], z[:, 0]))
            gy.append(CONDITIONS.index(r["condition"]))
            groups.append(r["group_id"])
    gate.fit(gx, gy, groups, select=not a.fixed_gate)
    gate.save(out / "gate.pkl")
    save(
        out / "gate_selection.json",
        dict(
            selected=gate.selected,
            cv=gate.cv_results,
            labels="training source classes only",
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allpred = []
    allmetrics = []
    for seed in a.seeds:
        a.active_seed = seed
        seed_all(seed)
        folder = out / f"seed{seed}"
        folder.mkdir()
        paths = train_models(train, a, device, folder)
        d, m = evaluate(paths, evaluation, gate, a, device, folder)
        allpred.append(d)
        allmetrics.extend(m)
    pd.concat(allpred).to_csv(
        out / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    d = pd.DataFrame(allmetrics)
    d.to_csv(out / "metrics_by_seed.csv", index=False)
    d.groupby(["model", "gate"]).agg(
        mae=("mae", "mean"),
        mae_sd=("mae", "std"),
        rmse=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
    ).reset_index().to_csv(out / "summary.csv", index=False)
    for r in data:
        if sha256(r["path"]) != r["sha256"]:
            raise RuntimeError("Source input changed during execution.")
    save(
        out / "complete.json",
        dict(
            passed=True,
            source_files_unchanged=len(data),
            seeds=a.seeds,
            prediction_rows=sum(len(d) for d in allpred),
            new_protocol=True,
        ),
    )
    print(d.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
