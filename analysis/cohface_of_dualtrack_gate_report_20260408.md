# COHFACE raw-OF plus OF-bridge assistant gate

Date: 2026-04-08

Run:
- `/home/gongjae/Projects/resPyre/results/20260408_of_dualtrack_gate/cohface_of_dualtrack_gate`

## Goal

Keep raw `OF` as the waveform-driving primary observation while using
`OF_bridge` as an output-only rate assistant.

## Compared methods

- `of_farneback__parh_ossm`
- `assist_ofbridge_of__parh_ossm`
- `of_disp_bridge__parh_ossm`

## Result

`assist_ofbridge_of__parh_ossm` was effectively identical to raw
`of_farneback__parh_ossm` for T4 and slightly worse for T3.

### raw OF PARH -> OF dual-track PARH

- waveform:
  - `CCC 0.785 -> 0.785`
  - `MAE 0.525 -> 0.525`
  - `DTW 0.382 -> 0.382`
- rate:
  - `MAE 0.670 -> 0.675`
  - `RMSE 0.840 -> 0.845`
  - `PearsonR 0.720 -> 0.710`

## Interpretation

The current assistant gate did not activate in a materially useful way. Raw
`OF` and the dual-track output were effectively the same scaffold, so this
does not yet realize the intended “raw waveform plus bridge rate” split.

## Verdict

`no-go`.

Do not promote this as a live family. If revisited later, the next attempt
should modify how assistant evidence enters the rate output rather than merely
reusing the current assistant gate with a different upstream signal.
