# Observation Family Semantics Lock

Date: 2026-04-07

## Purpose

This note fixes the intended semantic role of each observation family used in
PARH-OSSM so the paper, EDA, preprocessing policy, and state-space observation
design all talk about the same thing.

The central rule is:

- the extractor output is not "the respiration waveform"
- it is a family-specific camera motion proxy
- each family sees a different view of the same latent respiratory motion

Accordingly, PARH should not force one observation interpretation on all
families.

## Code lock

Family semantics are no longer only narrative. They are now mirrored in code by
the registry at:

- `components/observations/semantics.py`

This registry is the canonical place for:

- observation domain
- primary information type
- nuisance risk
- default inference-path mode
- whether rescue/helper-trust/family-confidence are even allowed

Future family-specific patches should start from that registry instead of
adding more string-matching rules across the filter.

## Generative chain

The paper should explain the signal path in this order:

1. physiology
2. camera projection and nuisance
3. family-specific proxy extraction
4. family-specific preprocessing
5. family-specific observation model
6. shared latent respiratory decomposition
7. dual outputs

Formally:

`physiology -> chest motion -> camera/view nuisance -> extractor family proxy -> observation model -> latent state -> z_osc / z_full`

## Shared latent state

The shared latent state remains:

`x_t = [h_c^(1), h_s^(1), h_c^(2), h_s^(2), b, b_dot, r, r_dot]`

Interpretation:

- `h^(1), h^(2)`: oscillatory respiratory drive and harmonic morphology
- `b`: slow baseline / projection drift
- `r`: aperiodic respiratory residual

This latent state is shared across all observation families.

What changes by family is the observation meaning of the same latent state.

## Family semantics

### OF-Farneback

Implementation source:
- `OF()` in `components/observations/motion.py`
- signal = median vertical component of dense optical flow between adjacent
  chest ROI frames

What it is closest to:

- a signed oscillatory motion / velocity-like proxy
- strong respiratory periodic content
- relatively weak absolute baseline semantics

Typical strengths:

- good oscillatory support
- good local phase/frequency evidence
- strong rate utility

Typical distortions:

- velocity-like phase shift relative to displacement truth
- weak direct amplitude semantics
- sensitive to optical-flow failures and projection changes

Model handling rule:

- treat OF mainly as an oscillatory view
- allow phase/lag-aware harmonic visibility
- keep baseline/residual observation gain conservative
- allow helper-path support and output-side rate refinement
- do not rely on OF to carry baseline amplitude directly

Current code state:

- light observation path
- selective internal frequency rescue allowed
- OF-only output-side helper rate blend promoted

Future target:

- explicit velocity-like harmonic observation row, not only a generic phase lag
- OF should ultimately inform oscillatory dynamics or helper evidence more than
  direct full-waveform reconstruction unless a velocity-to-displacement bridge
  is explicitly validated

Current lock after latest gate:

- direct warm-up OF velocity-row calibration failed badly on the COHFACE gate
  subset
- a fit-quality safety gate now disables such calibration when the warm-up fit
  is weak
- until a better formulation exists, OF should remain helper-heavy rather than
  observation-row-heavy
- helper-trust heuristics do not currently rescue this family:
  - direct helper-trust-driven `q_dyn` suppression harms OF rate
  - rescue-only helper trust is safe but inert
  - rescue-only helper trust plus relaxed rescue improves some OF rate metrics
    but harms waveform enough to reject promotion
- the next OF step should therefore target observation semantics rather than
  another helper-trust gate
- a direct single-family OF-to-displacement bridge has now produced the first
  strongly positive OF-only gate result:
  - raw `OF` remained better only as an unfiltered base waveform proxy
  - `OF bridge` substantially improved both T3 and T4 for `KFstd` and PARH on
    the 12-trial COHFACE gate subset
  - this bridge is therefore the first OF observation-semantics step promoted
    to full-dataset validation
- full 160-trial COHFACE validation is now completed:
  - `OF_bridge` substantially improves OF-family T3 for `KFstd` and PARH
  - `OF_bridge` slightly improves OF-family `waveform_MAE` / `DTW` for PARH
  - but `OF_bridge` does not improve OF-family waveform `CCC`
  - promotion decision: keep raw `OF` and add `OF_bridge` as a new observation
    family rather than replacing raw `OF`

### DoF

Implementation source:

- `DoF()` in `components/observations/motion.py`
- signal = thresholded count of large frame-to-frame pixel changes

What it is closest to:

- thresholded motion energy
- event-like motion magnitude proxy

Typical strengths:

- simple
- sensitive to large respiratory motion and bursts

Typical distortions:

- coarse amplitude semantics
- saturation and threshold artefacts
- weak waveform fidelity
- contamination by non-respiratory motion energy

Model handling rule:

- do not treat DoF as a clean displacement waveform
- keep robust observation-noise adaptation strong
- use DoF cautiously for T4 waveform claims
- interpret DoF as a noisy motion-energy family that may still help rate in
  some segments

Current code state:

- legacy preprocessing retained
- no promoted light-path routing
- no promoted observation calibration

Future target:

- add explicit nuisance handling before claiming waveform competitiveness
- likely requires a nuisance state or family-specific contamination term

### Profile1D linear

Implementation source:

- `profile1D()` in `components/observations/motion.py`
- signal = frame-to-frame shift surrogate from 1D vertical profile
  cross-correlation using linear interpolation

What it is closest to:

- displacement-like chest shift surrogate
- relatively smooth fundamental-dominant view

Typical strengths:

- useful oscillatory track when clean
- interpretable displacement-style proxy

Typical distortions:

- lag and interpolation bias
- weaker harmonic morphology than quad/cubic
- scale instability

Model handling rule:

- treat as displacement-like with modest harmonic complexity
- fundamental should dominate
- second harmonic visibility should stay conservative
- allow selective frequency rescue because gate evidence was positive

Current code state:

- legacy preprocessing retained
- selective freq rescue allowed
- observation calibration not promoted

Future target:

- bounded lag-aware harmonic observation row if it passes no-harm gates

### Profile1D quadratic / cubic

Implementation source:

- `profile1D()` in `components/observations/motion.py`
- same profile-displacement surrogate with higher-order interpolation

What it is closest to:

- displacement-like respiratory morphology proxy
- stronger harmonic visibility than P1D-linear

Typical strengths:

- best chance of capturing inhale/exhale asymmetry in a single-family proxy
- strong waveform fidelity on clean clips

Typical distortions:

- interpolation-induced harmonic reshaping
- small but important lag mismatch
- family-specific amplitude scaling

Model handling rule:

- these families should be the main target for harmonic visibility calibration
- allow stronger `h2` visibility than `h1`-only families
- keep calibration bounded and family-restricted

Current code state:

- legacy preprocessing retained
- selective `obs_cal_v5` promoted only for quad/cub
- warm-up bounded family-phase auxiliary calibration is currently the best
  observation-side improvement path

Future target:

- richer but still bounded harmonic observation semantics if gates stay
  positive

Current live refinement after latest COHFACE full rerun:

- a narrow family-confidence policy is now promoted for `P1D_quad/cub`
- activation requires excellent warm-up displacement fit
- the effect is intentionally small:
  - slightly lower T3 absolute error on the strongest profile families
  - near-zero effect on other families
  - no broad observation-policy change outside the target families

## Current redesign consequence

Recent gate evidence fixes two boundaries:

- residual-release heuristics alone are not enough; they can change `q_osc`
  without producing useful T3/T4 gains
- the next live redesign should therefore focus on:
  - OF-specific velocity semantics
  - stronger residual identifiability
  - not more generic preprocessing or generic `q_osc` heuristics

## Preprocessing interpretation

Preprocessing is not a generic nuisance-cleaning block.
It is part of the observation semantics.

Therefore the paper must show, family by family:

- raw signal
- detrended signal
- bandpassed signal
- current inference-path signal
- helper-path signal

The purpose is to answer:

- what does preprocessing actually fix
- what distortion remains after preprocessing
- which family can tolerate aggressive band-limiting
- which family loses waveform semantics under the same preprocessing

## Raw and preprocessed signal meaning by family

The paper should explicitly distinguish:

- raw extractor output
- current inference-path signal
- current helper-path signal

These are not interchangeable.

### OF-Farneback

Raw extractor output:

- frame-to-frame median vertical optical flow
- transition-domain quantity
- closest to signed chest-motion velocity proxy

After current inference preprocessing:

- light low-pass and robust centering/scaling preserve the oscillatory shape
- low-frequency drift is reduced, but baseline meaning remains weak

After current helper preprocessing:

- band-limited oscillatory evidence
- strongest for local support and rate evidence
- not suitable as the sole waveform observation because baseline and residual
  semantics are suppressed

### DoF

Raw extractor output:

- thresholded count of large inter-frame pixel changes
- nonnegative motion-energy-like transition signal
- event-sensitive and easily contaminated

After current inference preprocessing:

- still closer to motion energy than to displacement
- smoothing helps stability, but amplitude semantics remain weak

After current helper preprocessing:

- can expose periodic bursts but may over-regularise the signal
- does not solve threshold contamination or nuisance mixing

### Profile1D linear

Raw extractor output:

- frame-to-frame profile-shift surrogate from 1D chest profile
- closest to displacement-like motion with modest interpolation bias

After current inference preprocessing:

- preserves displacement-style oscillatory content
- still subject to scale mismatch and small lag bias

After current helper preprocessing:

- useful for frequency evidence
- can remove some morphology needed for full waveform fidelity

### Profile1D quadratic / cubic

Raw extractor output:

- same displacement-style surrogate as P1D-linear
- higher-order interpolation sharpens peak location
- stronger harmonic content and morphology sensitivity

After current inference preprocessing:

- waveform semantics are largely preserved
- family-specific lag and harmonic reshaping remain

After current helper preprocessing:

- good helper evidence for oscillatory support
- can flatten some morphology if used directly as waveform observation

## Model-link rule by family

The shared latent state is fixed, but the observation role differs by family.

### OF-Farneback

- primary role in the model: oscillatory support and rate evidence
- secondary role: limited waveform reconstruction through shared latent state
- current handling:
  - light observation path
  - conservative baseline/residual visibility
  - output-only helper blend for `track_hz`
- intended final observation semantics:
  - velocity-like harmonic observation row with bounded phase lag

### DoF

- primary role in the model: motion-energy cue under uncertainty
- secondary role: only cautious contribution to waveform
- current handling:
  - legacy preprocessing
  - strong reliance on adaptive noise handling
- intended final observation semantics:
  - nuisance-aware motion-energy observation, possibly with an explicit
    contamination term or nuisance state

### Profile1D linear

- primary role in the model: fundamental displacement-like oscillatory proxy
- secondary role: moderate waveform reconstruction
- current handling:
  - legacy preprocessing
  - selective frequency rescue only
- intended final observation semantics:
  - bounded lag-aware fundamental-dominant observation row

### Profile1D quadratic / cubic

- primary role in the model: displacement-like morphology proxy with stronger
  harmonic visibility
- secondary role: best single-family target for waveform-oriented calibration
- current handling:
  - legacy preprocessing
  - selective bounded `obs_cal_v5`
- intended final observation semantics:
  - bounded harmonic-visibility observation row with small lag allowance and
    conservative auxiliary visibility

## Observation model design rule

The intended observation equation is family-specific:

`y_t^(m) = c_m + H_m(theta_m) x_t + v_t^(m)`

where:

- `m` is the family
- `theta_m` contains bounded family-specific calibration parameters
- `x_t` is the shared latent respiratory state

Design requirement:

- same latent state
- different observation semantics by family
- preserve conditional exact linearity after warm-up calibration

## Paper writing rule

The manuscript should never say:

- "all methods observe the same respiratory waveform"
- "the model simply denoises the input"
- "preprocessing removes the family differences"

The manuscript should say:

- each family is a different camera-derived proxy of respiratory motion
- preprocessing changes the observable semantics differently by family
- PARH uses a shared physiology-aligned latent state with family-aware
  observation handling

## Current implementation lock

As of 2026-04-07:

- OF:
  - light observation path
  - selective internal freq rescue
  - conservative output-only helper-rate blend
- DoF:
  - legacy path only
  - no promoted observation calibration
- P1D-linear:
  - legacy path
  - selective freq rescue
- P1D-quad/cub:
  - legacy path
  - promoted bounded harmonic-only `obs_cal_v7`

## Next implementation priorities

1. keep shared latent decomposition fixed
2. continue family-aware observation handling instead of global calibration
3. improve OF rate via output-side helper integration, not waveform-side
   regression
4. avoid broad calibration promotion unless gate-positive
5. treat DoF as the most likely nuisance-limited family
