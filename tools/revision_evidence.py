"""Recompute calibration, equal-input noise and published-value diagnostics.

This does not reconstruct undocumented historical calibration or change CSVs.
Run from the repository root; all artifacts go to --output.
"""

from pathlib import Path
import sys, argparse, json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from utils.preprocessing import PHICalculator
from utils.revision_data import read_manifest, sha256
from utils.metrics import calculate_rmse, calculate_mae, calculate_nasa_score


def save(p, x):
    Path(p).write_text(
        json.dumps(x, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def fit_pair(force, resistance):
    f = np.asarray(force, float)
    y = np.asarray(resistance, float)
    good = (f > 0) & (y > 0) & (y < 1e12) & np.isfinite(f) & np.isfinite(y)
    f = f[good]
    y = y[good]
    if len(f) < 3:
        return dict(n=len(f), status="insufficient")
    x = np.column_stack([f ** (-1 / 3), np.ones(len(f))])
    ab = np.linalg.lstsq(x, y, rcond=None)[0]
    yp = x @ ab
    return dict(
        n=len(f),
        alpha=float(ab[0]),
        beta=float(ab[1]),
        r2=float(1 - ((y - yp) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-15)),
        condition_number=float(np.linalg.cond(x)),
        units="as_stored",
    )


def table_metrics(path, out):
    d = pd.read_csv(path)
    error = abs(d.prediction - d.target).to_numpy()
    sigma = d.sigma.to_numpy()
    r, p = pearsonr(sigma, error)
    sp, spp = spearmanr(sigma, error)
    # Even if all three printed columns were rounded to whole cycles, this
    # bound rules out changes far larger than the rounding uncertainty.
    bound = 2 * np.sqrt(len(d)) * 0.5 / np.linalg.norm(
        sigma - sigma.mean()
    ) + 2 * np.sqrt(len(d)) * 1.0 / np.linalg.norm(error - error.mean())
    result = dict(
        n=len(d),
        mae=float(calculate_mae(d.prediction.to_numpy(), d.target.to_numpy())),
        rmse=float(calculate_rmse(d.prediction.to_numpy(), d.target.to_numpy())),
        scaled_log_nasa=float(
            calculate_nasa_score(d.prediction.to_numpy(), d.target.to_numpy())
        ),
        pearson_sigma_absolute_error=float(r),
        pearson_p_descriptive=float(p),
        spearman=float(sp),
        spearman_p_descriptive=float(spp),
        maximum_pearson_change_from_whole_cycle_rounding_bound=float(bound),
        metric_definition="cycle-space values; NASA errors/10000 then log10(1+sumexp); Pearson acrosssameobservations",
        source_sha256=sha256(path),
    )
    save(out / "printed_table_recalculation.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest")
    p.add_argument("--output", required=True)
    p.add_argument("--static-fsr")
    p.add_argument("--table")
    p.add_argument("--historical-mc")
    p.add_argument("--cut-file")
    p.add_argument("--cut-rul", type=float, default=73900.0)
    p.add_argument("--noise-repeats", type=int, default=10)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    cal = []
    noise = []
    ident = []
    source = []
    if a.manifest:
        records, overlap = read_manifest(a.manifest, allow_overlap=True)
        phi = PHICalculator()
        zero_beta = PHICalculator(beta=0.0)
        for r in records:
            z = r["array"]
            source.append(
                dict(record_id=r["record_id"], path=r["path"], sha256=r["sha256"])
            )
            if r["split"] == "train":
                early = z[(z[:, 0] >= 0) & (z[:, 0] <= 20000)]
                cal.append(
                    dict(
                        record_id=r["record_id"],
                        condition=r["condition"],
                        region="cycles0to20000",
                        **fit_pair(early[:, 2], early[:, 1])
                    )
                )
            if r["split"] != "val":
                continue
            R, F = z[:, 1], z[:, 2]
            clean = phi.calculate_phi(R, F)[0]
            r_only = np.clip((R - R[:100].mean()) / 30, 0, 1)
            no_beta = zero_beta.calculate_phi(R, F)[0]
            ident.append(
                dict(
                    record_id=r["record_id"],
                    phi_vs_residualR_only_max_abs=float(abs(clean - r_only).max()),
                    beta_removal_max_abs=float(abs(clean - no_beta).max()),
                    raw_residual_positive_fraction=float(
                        (R - (2.34 * F ** (-1 / 3) + 0.133) > 0).mean()
                    ),
                    phi_zero_fraction=float((clean == 0).mean()),
                    phi_one_fraction=float((clean == 1).mean()),
                )
            )
            for level in [0.005, 0.01, 0.03]:
                for seed in range(a.noise_repeats):
                    rng = np.random.default_rng(
                        20261024 + int(r["sha256"][:8], 16) + seed
                    )
                    noisy = (
                        np.column_stack([R, F])
                        + rng.normal(size=(len(R), 2))
                        * abs(z[:100, 1:3].mean(0))
                        * level
                    )
                    if (noisy[:, 1] <= 0).any():
                        raise ValueError(
                            "Noise made force nonpositive; revise declared scenario."
                        )
                    changed = phi.calculate_phi(noisy[:, 0], noisy[:, 1])[0]
                    for channel, c, n in [
                        ("res", R, noisy[:, 0]),
                        ("PHI", clean, changed),
                    ]:
                        rc = abs(pearsonr(c, z[:, 0]).statistic)
                        rn = abs(pearsonr(n, z[:, 0]).statistic)
                        noise.append(
                            dict(
                                record_id=r["record_id"],
                                level=level,
                                seed=seed,
                                channel=channel,
                                clean_abs_r=rc,
                                noisy_abs_r=rn,
                                correlation_drop=rc - rn,
                                scenario="synthetic_equal_input_sensor_noise",
                            )
                        )
        pd.DataFrame(cal).to_csv(out / "initial_region_fits.csv", index=False)
        pd.DataFrame(ident).to_csv(out / "phi_identifiability.csv", index=False)
        d = pd.DataFrame(noise)
        d.to_csv(out / "equal_input_noise.csv", index=False)
        summary = (
            d.groupby(["level", "channel"])
            .agg(
                clean_r=("clean_abs_r", "mean"),
                noisy_r=("noisy_abs_r", "mean"),
                drop=("correlation_drop", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(out / "equal_input_noise_summary.csv", index=False)
        ratios = []
        for level in [0.005, 0.01, 0.03]:
            q = summary[summary.level == level].set_index("channel")["drop"]
            ratio = (
                float(q["res"] / q["PHI"])
                if q["res"] > 1e-8 and q["PHI"] > 1e-8
                else None
            )
            ratios.append(
                dict(
                    level=level,
                    raw_drop=float(q["res"]),
                    phi_drop=float(q["PHI"]),
                    drop_ratio_raw_over_phi=ratio,
                )
            )
        save(
            out / "noise_ratio.json",
            dict(
                definition="mean absolute-correlation drop(raw)/drop(PHI), same inputnoise; undefined for nonpositive/negligibledrop",
                repeats=a.noise_repeats,
                ratios=ratios,
                not_historical12x_reconstruction=True,
            ),
        )
    if a.static_fsr:
        d = pd.read_csv(a.static_fsr)
        rows = []
        for col, z in d.groupby("source_column_number"):
            rows.append(
                dict(
                    column=str(col),
                    **fit_pair(z.force_as_stored, z.resistance_as_stored)
                )
            )
        rows.append(
            dict(column="pooled", **fit_pair(d.force_as_stored, d.resistance_as_stored))
        )
        pd.DataFrame(rows).to_csv(out / "static_fsr_fits.csv", index=False)
    if a.table:
        table_metrics(a.table, out)
    if a.historical_mc:
        z = np.load(a.historical_mc)
        s = z["predictions_std"].ravel()
        e = z["abs_errors"].ravel()
        r, pv = pearsonr(s, e)
        save(
            out / "historical_mc_correlation.json",
            dict(
                n=len(s),
                pearson=float(r),
                p_descriptive=float(pv),
                source_sha256=sha256(a.historical_mc),
                scope="separatelysavedwindows; notsameunitas9tableobservations",
            ),
        )
    if a.cut_file:
        d = pd.read_csv(a.cut_file)
        end = d.cumulative_contact.iloc[-1]
        observed = end - a.cut_rul
        matches = d.index[np.isclose(d.RUL, a.cut_rul)].tolist()
        save(
            out / "cutoff_inverse.json",
            dict(
                file=Path(a.cut_file).name,
                eol_cycle=float(end),
                paper_target_RUL=a.cut_rul,
                inferred_observed_cycle=float(observed),
                cycle_fraction=float(observed / end),
                matching_row_indices=matches,
                historical_identity_confirmed=False,
                scope="conditional inverse under same-file mapping; notthe original selectionreason",
            ),
        )
    for r in source:
        if sha256(r["path"]) != r["sha256"]:
            raise RuntimeError("Source input changed.")
    save(
        out / "protocol.json",
        dict(
            script_sha256=sha256(__file__),
            sources=source,
            input_csv_modified=False,
            calibration="OLS y=alpha*F^(-1/3)+beta; use recorded units; notfittooldtargetcoefficients",
            noise="same synthetic R/F input for bothrepresentations; no modelperformance-basedvalidationchanges",
            scope="newdiagnostics, historicalvalues and possible explanations remain separate",
        ),
    )
    print("EVIDENCE COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
