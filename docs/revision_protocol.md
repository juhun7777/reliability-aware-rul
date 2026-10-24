# Revised training and condition-similarity gate

This is a revised development protocol. It is not a reconstruction of an
undocumented historical execution that produced a manuscript table.

## Fixed data and observation identity

`train_revision.py --manifest ... --output ...` consumes an explicit CSV with
`record_id,path,split,condition,group_id`. Relative paths resolve against the
manifest directory. `condition` is one of `0A,0.5A,1.0A` for training; evaluation
folders can have arbitrary condition names. They are never supplied to the gate.
`group_id` must keep original trajectories and their augmented descendants
together. A group may deliberately block several related originals together.
It is not inferred from the filename or current label.

The bundled example uses file hashes because original specimen identities are
not supplied with the public sample. These are **file-level demonstration groups**,
not evidence of independent specimens. Replace them with original-parent groups
for actual experiments. Exact content duplicates share one demonstration group.

CSV columns are `cumulative_contact,res,force,RUL`. Missing RUL is an error in
the revision entrypoint; row counts are never silently treated as cycle units.
The code does not infer whether RUL=0 corresponds to physical failure or test
termination. That must be documented from the experiment.

The default observation fractions are 0.7, 0.8, 0.9. They choose offline
evaluation rows; neither the fraction nor final lifetime is a model input.
Raw sensor values, row count, endpoint and RUL are never changed.

Overlapping train/evaluation parent groups are rejected unless
`--allow-development-overlap` is explicitly supplied. Such replay records
the overlap in `config.json` and remains a development comparison.

## Inputs and PHI

The five network channels are:

1. PHI = clip((abs(R - (alpha * F**(-1/3) + beta)) - mean(first100 residuals))/30,0,1)
2. recorded resistance
3. recorded force
4. (R - mean(first100 R)) / mean(first100 R)
5. measured cumulative_contact at the observation / 380000, repeated across the window

The first four channels are standardized using training files only.
The PHI ablation zeros the standardized PHI channel and keeps the same network
size. These inputs do not contain the STFT classifier's 32 statistics.
`relative_res` is an initial-relative change, **not dR/dt**.

Default alpha=2.34 and beta=0.133 are legacy configuration constants, not newly
estimated calibration results. If resistance uses unit U_R and force U_F,
alpha has units U_R * U_F**(1/3), and beta has units U_R.
`tools/revision_evidence.py` fits the stated initial 0--20000 cycle region and
static FSR data with recorded units. It never searches a unit conversion or
gain to force agreement with old coefficients. Multiplicative unit conversions
alone cannot change a linear model's R² when applied consistently.

The legacy `PHICalculator` now defaults to an initial 100-row baseline.
`baseline_method='legacy_full_min'` preserves the former whole-trajectory
minimum for explicit historical comparison. Those protocols differ.

## Gate

`STFTRateGate` subtracts the first 100 observed resistance samples' mean and
computes a Hann STFT with segment 256, overlap 128, no boundary extension,
and no padding. The CSV must have a uniform increasing cycle grid. Frequency
is indexed per CSV row; the code does not invent a physical acquisition Hz.
At 100 cycles per row, 256 rows cover 25,600 cycles, not 2.56 seconds.

The primary DC representation summarizes amplitude mean, std, quantiles and
time slope. Amplitude summaries are divided by observed cumulative cycles to
account for exposure. Initial mean resistance and observed cycles are also
inputs. No final endpoint/RUL/current label is needed for inference.

The DC component equals a Hann-weighted moving mean. A successful DC-only
gate is evidence for amplitude/trend similarity routing, **not proof that
nonzero frequency content is necessary**. `--gate-bins nonzero_control` adds
bins1/2 under the same normalization for a separate control.

Regularization and classifier are chosen by train-only stratified original-group
CV. Scaling is fit inside each CV training fold. The selected pipeline is refit
on training prefixes only. `--fixed-gate` instead uses the declared SVM C=1
configuration without a setting search. Gamma=1 is the revised primary soft
combination; hard and uniform use the same expert outputs.

The historical STFT+MLP class remains available. Its window features are now
split by file/parent before fitting the scaler. An explicit original-parent
group map is needed to keep augmented siblings together across file names.
Best weights are cloned, not kept as an alias of live training tensors.
The legacy0.65/0.35 block multipliers occur before per-column standardization,
which cancels those constant scalings. They must not be interpreted as actual
65percent/35percent post-scaling feature contributions or validated attention
weights. A controlled block-weight experiment must apply weights after scaling.

## Regression models and selection

Pooled: all selected training conditions train one Transformer.
Independent experts: each of three Transformers sees its own condition.
Shared experts: a common Transformer encoder sees all conditions, while the
known source condition supervises its own output head during training. No
condition identifier is supplied at inference; the gate combines the heads.

Each seed chooses an internal original group per class for epoch selection,
fits scalers only on that internal training set, and selects minimum cycle-space
RMSE (at most50epochs, patience12). It then refits all training files for the
selected epoch count, replaying the selected learning-rate schedule with a
full-training-only scaler. External val is never used for this selection.

Networks use 200-row windows, stride100, the last40% of training trajectories,
3 Transformer layers, width128, 4 attention heads, dropout0.1, AdamW lr5e-4,
weight decay1e-4, Huber delta0.5 on RUL/100000. The fixed 50:50 pooled/expert
combination is a control, not an optimized validation mixing ratio.

## Exports and uncertainty

Every prediction records record ID, parent group, exact observation row/cycle,
all expert outputs and weights, target, error, seed, model and gate mode.
Each checkpoint includes a config, scaler, selected epoch and train/calibration
IDs. Input SHA256 hashes are recorded and rechecked after execution.

Default `--mc-samples 0` produces deterministic predictions and leaves sigma
missing. `--mc-samples 30` enables explicit dropout modules while keeping other
layers in evaluation mode, fixes the gate per observation, saves all draws,
and reports their mean and standard deviation in **cycle space**. Module modes
are restored after inference. This is an experimental uncertainty definition,
not automatically a calibrated prediction interval or historical rho=0.68.
The MC seed is `seed+100000`, independent of optimizer RNG history. Each
weighted draw is clipped at zero before computing ensemble mean/std.
Raw component draws and their means are saved separately from deterministic
component outputs, together with the fraction of draws clipped. Reconstruct
the ensemble from those draws when clipping makes means non-linear.

The legacy Bi-LSTM path has a different MC policy that enables the whole model's
training mode. Compare uncertainty policies explicitly; do not mix their sigma.

The reported NASA diagnostic is log10(1 + sum(exp(asymmetric error/10000)-1)).
It is not the unscaled NASA challenge total. Perfect predictions score zero.

## Train-calibrated mixing control

`tools/calibrate_mixture.py` uses the saved internal calibration groups and
their **inner**, not full-training, checkpoints to choose the pooled fraction
from0,.25,.5,.75,1. An inner gate is refit without those calibration groups.
All choices are saved before applying them to val. Group-average MAE is used
so augmented observations do not act like independent calibration specimens.
The calibration set already selected epoch count, so this remains a small-sample
development setting, not an independent nested generalization estimate. Compare
its deterministic output against deterministic pooled/fixed50 outputs. A poor
result must be retained rather than changing the grid after seeing val.

## Historical questions

Use `tools/revision_evidence.py` for initial-region/static calibration,
equal-input Gaussian sensor-noise diagnostics, printed-table RMSE/NASA and
sigma/error correlation, historical MC-array correlation, and conditional
cutoff inversion. All noise experiments are synthetic diagnostics with fixed
labels. They cannot restore an undocumented gain history or authenticate
an old figure's statistic. See `revision_questions.md` for interpretation.
