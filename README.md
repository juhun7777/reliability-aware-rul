# Reliability-Aware RUL Prognostics for Semiconductor Test Socket Micro-Pins

Research code for condition-specific RUL prediction from resistance and force.
The repository contains the original Bi-LSTM implementation and a revised,
explicit-manifest workflow with pooled, independent-expert and shared-encoder
Transformer controls.

The revised workflow is a **new development protocol**. Running its defaults
is not a claim to reproduce the historical manuscript tables or calibration.
The original README is preserved in [docs/legacy_readme.md](docs/legacy_readme.md).

## Revised workflow

Use Python3.10+ and the dependencies in `requirements.txt` (tested with
Python3.12, PyTorch2.7.1, NumPy1.26.4, pandas2.2.2, SciPy1.16.3,
scikit-learn1.5.1). No network download occurs during training.

```text
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The bundled18CSV sample is for code verification. Its manifest uses file-level
hash groups because authenticated original-parent identities are not included.
It is not a claim of18independent measured pins or independent unseen-current
validation. For a short execution check:

```text
python train_revision.py --manifest examples/sample_manifest.csv --output results/demo --models shared --seeds 42 --epochs 2 --fixed-gate
```

For actual experiments, supply an explicit manifest with original-parent groups:

```csv
record_id,path,split,condition,group_id
train_01,/path/train/0A/pin01.csv,train,0A,original_01
train_02,/path/train/0.5A/pin02.csv,train,0.5A,original_02
val_01,/path/val/0.3A/pin03.csv,val,0.3A,original_03
```

This is a column-format illustration, not a complete trainable manifest. Each
training condition needs multiple original groups for internal selection.
CSV columns are `cumulative_contact,res,force,RUL`; units must be documented by
the experiment. Evaluation folder names are never used as gate inputs.

```text
python train_revision.py --manifest /path/manifest.csv --output results/revision --models all --seeds 42 43 44 --mc-samples 30
```

The output directory must be new. Epoch and gate settings are selected within
training groups; fixed val observations are70/80/90percent by default. Exact
observation rows/cycles, seed, all component predictions/weights, errors, MC
draws, scalers, histories, internal split IDs and input SHA256 hashes are saved.
Overlapping train/evaluation parent groups require the explicit
`--allow-development-overlap` option and are recorded as development replay.
Raw measurements, RUL, endpoints and file names are never rewritten.

## Condition similarity and model controls

- `STFTRateGate` compares observed low-frequency amplitude/trend features after
  accounting for elapsed cycles. Training classes are0A,0.5A,1.0A. New signals
  receive continuous expert weights, not a fabricated new current class.
- The primary gate uses the STFT **DC** component. This is equivalent to a
  Hann-weighted moving mean; its effect does not establish that nonzero
  frequency content is necessary. `--gate-bins nonzero_control` provides a
  separate bins0/1/2 control with the same exposure normalization.
- `--models all` trains a pooled model, three independent condition experts,
  and a common Transformer encoder with three condition-specific output heads.
  Soft/hard/uniform combinations use the same expert outputs. A fixed50:50
  pooled/expert blend is also exported; its ratio is not tuned on val.
- `--no-phi` zeros the standardized PHI channel without reducing model capacity.
  Legacy alpha/beta remain configurable constants; do not interpret them as a
  newly recovered physical calibration.

See [the full revision protocol](docs/revision_protocol.md) for scaling,
checkpoint selection, MC policy, limits and the distinction between training
source classes and measured intermediate currents.

Replay saved checkpoints and the independent `seed+100000` MC seed:

```text
python tools/replay_revision.py --run results/revision --manifest /path/manifest.csv --output results/replay
```

To test choosing the pooled/expert fraction using only training-internal
calibration groups, without re-training or changing val:

```text
python tools/calibrate_mixture.py --run results/revision --manifest /path/manifest.csv --output results/calibrated_mixture
```

All candidate fractions and frozen selections are exported before val scoring;
the result may be worse than the fixed50:50 control.

Compare DC versus added bins using the same saved regression predictions and
the same MC draws (requires an `--models all` run):

```text
python tools/compare_gate_bins.py --run results/revision --manifest /path/manifest.csv --output results/gate_bins
```

## Historical Bi-LSTM path

```text
python train.py
```

This remains a separate protocol. Repairs include removing the0A-specific test
file replacement, fixing the observation fraction to0.8, using measured cycles,
separating STFT groups before scaling, restoring the best epoch in memory, and
exporting per-file predictions/config/MC draws. PHI now defaults to an initial
100-row baseline; `legacy_full_min` is an explicit calculator option for old
comparisons. These changes can alter results; they must not be called an exact
historical replay. The sample file split alone cannot establish original-parent
independence without a supplied group map.

## Recalculate revision evidence

```text
python tools/revision_evidence.py --manifest /path/manifest.csv --output results/evidence --table /path/table_viii.csv --static-fsr /path/fsr_points.csv --historical-mc /path/mc_dropout_results.npz
```

Optional inputs can be omitted. The table schema includes `target,prediction,sigma`;
static FSR points use `source_column_number,force_as_stored,resistance_as_stored`.
The tool fits initial0--20000cycle calibration, compares PHI/raw using the same
synthetic sensor noise, recalculates printed metrics and uncertainty/error
correlations, and checks input preservation. These diagnostics do not invent
undocumented gains or recreate the historical12x/rho0.68claims.

See [revision questions and primary bibliography records](docs/revision_questions.md).
The scaled/log NASA diagnostic is explicitly named; perfect predictions score0.
A correlation between sigma and error does not by itself establish calibrated
prediction intervals.

## Authors and license

Juhun Lee, Kihyun Kim, Hyo-Young Kim. Tech University of Korea.
Research manuscript information should follow its current submitted/accepted
version; the legacy README reflects an earlier submission description.
MIT License. See [LICENSE](LICENSE).
