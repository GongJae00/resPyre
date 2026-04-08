# PARH-OSSM Evidence Flow Lock

## Purpose

This file locks how evidence should appear in the paper from early-stage signal
inspection to final quantitative claims.

It exists to prevent a common failure mode:

- putting metrics first
- leaving observation semantics implicit
- only later explaining preprocessing and diagnostics as if they were side notes

For this paper, that order is wrong.

The paper must move in the same order as the scientific chain:

1. physiology
2. camera/projection distortion
3. extractor-family observation proxy
4. preprocessing effects
5. calibrated state-space modelling
6. output-specific evaluation
7. diagnostics and failure analysis

## Main-paper evidence order

### Section 2. Generative framing

Goal:

- explain what the hidden process is
- explain why the observation is distorted before it ever reaches the model

Main figure:

- `F1`: physiology -> projection -> proxy family -> calibration/helper -> OSSM decomposition -> `z_osc` / `z_full` / diagnostics

### Section 4. Methods

Goal:

- explain what is observed at each stage
- explain what preprocessing is allowed to do and what it cannot do

Main table:

- `T2`: model-component / observation / metric-routing map

Key wording to preserve:

- raw family signals are not interchangeable
- helper preprocessing is evidence extraction, not the final observation
- inference preprocessing is a modelling choice that can help some families and
  hurt others

### Results R1. Dataset and observation characterization

Goal:

- characterize regime and observation quality before reporting model metrics

Main figure:

- `F2`: dataset regime + observation/preprocessing characterization

Recommended panel structure:

- `F2A`: dataset regime summary (trial length / GT rate variability)
- `F2B`: family-level raw observation quality spread
- `F2C`: preprocessing-stage delta heatmap
- `F2D`: family-specific best-stage summary

Supplementary expansion:

- full stage heatmaps by metric
- raw vs current vs helper stage example overlays

### Results R2-R4. Model outputs

Goal:

- evaluate each output only for the job it was designed for

Tables:

- `T3`: rate metrics from `z_osc -> track_hz -> BPM`
- `T4`: waveform metrics from `signal_hat` / `z_full`

Rules:

- `T3` answers oscillatory tracking
- `T4` answers band-limited respiratory waveform fidelity
- do not use `T4` as a rate surrogate
- do not use `T3` to claim full waveform equivalence

### Results R5-R6. Diagnostics and ablation

Goal:

- explain why results change
- show that the model's internal semantics are auditable

Tables/figures:

- `T5`: intent-aligned ablation
- `T6`: calibration diagnostics
- `F5`: mechanism / calibration figure

This is where the paper should connect:

- observation calibration
- `q_obs`, `q_dyn`, `q_osc`
- residual/baseline energy
- strict vs relaxed calibration split

### Results R8-R9. Failure analysis

Goal:

- show what preprocessing and modelling still do not fix

Figure:

- `F6`: failure cases

Recommended failure-case categories:

- observation mismatch remains despite good rate tracking
- irregular structure leaks into the oscillator path
- family disagreement or weak visibility

## Stage-specific evidence policy

### Stage S0. Raw observation families

What to show:

- correlation to GT waveform / derivative
- spectral concentration
- failure modes by family

Best placement:

- R1 / `F2`

### Stage S1. Generic preprocessing

Question:

- what does a standard respiratory preprocessing stack actually fix?

What to show:

- raw -> detrend -> bandpass -> sign-align -> robust-z -> current preprocess -> helper preprocess

Best placement:

- R1 / `F2`
- supplementary full heatmaps

### Stage S2. Model-facing observation path

Question:

- after preprocessing, what mismatch is still left for the model?

What to show:

- family-aware observation-path choices
- why one global preprocess is not sufficient

Best placement:

- end of Methods
- opening of Model section

### Stage S3. Model outputs

Question:

- does the decomposition improve the right thing?

What to show:

- `z_osc` rate results
- `z_full` waveform results
- overlays by top/median/bottom cases

Best placement:

- `T3`, `T4`, `F3`, `F4`

### Stage S4. Diagnostics

Question:

- what is the model doing internally when it succeeds or fails?

What to show:

- `NIS`
- `pi_t`
- `lambda_t`
- `q_obs`, `q_dyn`, `q_osc`
- residual/baseline energy

Best placement:

- `T6`, `F5`

## Hard manuscript rule

Every main-paper quantitative claim must be explainable by an earlier evidence
stage.

Examples:

- if `T4` improves, the paper should already have shown which observation-path
  or calibration change made that plausible
- if `T3` improves but `T4` does not, the paper should already have explained
  why oscillatory tracking can improve while waveform fidelity remains limited
- if a family behaves differently from another family, `F2` or the observation
  characterization must already have shown that the two families are not
  observation-equivalent
