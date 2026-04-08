# COHFACE Residual-Release Gate Report (2026-04-07)

## Verdict

Residual-release redesign is still `no-go`.

The trigger direction is now better understood, but none of the tested
policies produced promotable T3/T4 gains on the 12-trial COHFACE gate subset.

## Tested policies

### `residual_release_v1`

Mechanism:

- `Q_OSC_OBS_WEIGHT=0.25`
- blends clean observation support directly into `q_osc`

Observed behavior:

- `obs_osc_support` was very high on clean COHFACE segments
- this pushed `q_osc` upward instead of opening the residual branch

Representative median `q_osc` shifts versus `full`:

- `OF`: `+0.0176`
- `P1D_linear`: `+0.0427`
- `P1D_quad`: `+0.0706`
- `P1D_cub`: `+0.0710`

Performance impact:

- essentially neutral or slightly worse
- therefore not promotable

### `residual_release_v2`

Mechanism:

- `Q_OSC_OBS_MODE=penalize_unexplained_v1`
- clean unexplained observation content can only reduce `q_osc`
- no direct extra `Q_aper` bonus

Representative median mechanism shifts versus `full`:

- `OF`: `q_osc -0.0603`, `obs_nonosc_need 0.319`
- `P1D_linear`: `q_osc -0.0291`, `obs_nonosc_need 0.094`
- `P1D_quad`: `q_osc -0.0020`, `obs_nonosc_need 0.009`
- `P1D_cub`: `q_osc -0.0021`, `obs_nonosc_need 0.010`

Performance deltas versus `full`:

- `OF` rate: `MAE +0.005`, `RMSE +0.005`, `r +0.000`
- `OF` waveform: `CCC -0.00028`, `MAE +0.00079`, `DTW +0.00037`
- `P1D_linear` rate: essentially unchanged
- `P1D_quad/cub`: unchanged

Interpretation:

- the trigger moved in the intended direction
- but the residual branch did not turn that change into meaningful T3/T4 gains

### `residual_release_v3`

Mechanism:

- same penalty as `v2`
- plus direct `Q_aper` bonus from clean unexplained observation need

Representative median mechanism shifts versus `full`:

- `OF`: `q_osc -0.0621`, `obs_nonosc_need 0.326`
- `P1D_linear`: `q_osc -0.0296`, `obs_nonosc_need 0.096`

Performance deltas versus `full`:

- `OF` rate: `MAE +0.010`, `RMSE +0.010`, `r +0.000`
- `OF` waveform: `CCC -0.00079`, `MAE +0.00201`, `DTW +0.00102`
- `DoF` rate: `MAE +0.085`, `RMSE +0.075`
- `P1D_quad/cub`: unchanged

Interpretation:

- stronger residual opening did not help
- in practice it slightly hurt `OF` and `DoF`

## Conclusion

The remaining bottleneck is not only the residual-release trigger.

Current evidence suggests:

- COHFACE clean segments often have very high oscillatory support
- lowering `q_osc` alone is not enough to improve waveform or rate
- directly boosting `Q_aper` also does not solve the gap

Therefore the next redesign should not focus on more `q_osc` heuristics alone.
The next residual step must make the residual branch more identifiable, for
example by:

- residual-specific observation semantics
- explicit event-like residual diagnostics
- stronger distinction between oscillatory mismatch and nuisance mismatch
