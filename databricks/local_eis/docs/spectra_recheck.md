# Re-check of the FAMOS spectra (order 2612030, all conditions, all segments)

Reason: the FAMOS Nyquist plot at 450 A looks zig-zagged below about 20 Hz, while the bench tool's CSV spectra are smooth.
Inputs: the FAMOS results that were delivered (`bronze/schedule.csv`, `bronze/gamry_sync.csv`, silver and gold) for 45, 60, 150 and 450 A.
Per-segment table: `figures/spectra_recheck/segment_spectrum_quality.csv`.

## 1. The impedance formula is not the problem

The checks were:
- duplicate frequencies;
- out-of-order steps;
- silver changing the low-frequency values compared with bronze.

None of these occurs. Z = K·A_U/A_s gives the right answer wherever the data window is right and the signal is clean. At 45 and 60 A all 67 segments are smooth, with a median deviation of 1.4–1.7 % from a smooth DRT curve.

## 2. One real computation error: misplaced low-frequency windows (now fixed)

The time window used for each frequency step comes from the Gamry DTA time stamps, which have 1 s resolution. `gamry_sync.align` checks only the **end** of each window. Some low-frequency windows therefore ran into the neighbouring step:
- 450 A, 0.19 Hz: the window starts 9.6 s too early;
- 450 A, 0.24 Hz: the window ends 6.9 s too early;
- there is overlap at 3–9 Hz as well.

Replaying the real schedule on a synthetic stepped sweep gives a median error of 0.9 % per point, and up to 4–5 % at 3–6 Hz.

**Fix:** `gamry_sync.confine_windows` and `separate_windows`, switched on by `window_confine=True`.
- Each window is put between two consecutive Gamry step ends, with a guard and half a period of settling.
- Overlapping windows are cut at their midpoint.
- In the simulation the error falls to 0 %.

Tests: `tests/test_window_confine.py`.
This fix needs the raw .DAT files. Re-run on Databricks with run mode `rerun`.

## 3. The main cause: low-frequency noise at the air outlet at 150 and 450 A

The table gives medians per third of the air path.

| | inlet | middle | outlet |
|---|---|---|---|
| 150 A: LF deviation from a smooth curve | 1.6 % | 2.8 % | 5.0 % |
| 450 A: LF deviation from a smooth curve | 2.3 % | 4.2 % | **10.1 %** |
| 450 A: LF points kept by the gates | 90 % | 90 % | **76 %** |
| 450 A: LF SNR | 7.7 dB | 8.0 dB | 6.4 dB |
| 450 A: LF THD | 4.1 % | 3.6 % | 6.1 % |

- The scatter grows steadily along the air path: r = 0.97–0.98 with flow position.
- It goes with THD (r = 0.84–0.91) and against SNR (r = −0.84).
- The outlet segments scatter **together** at each step (pairwise r = +0.98). They share a disturbance, so this is not random noise in each channel.

That pattern is what a cell that is **not stationary** at the outlet produces during the minute-long low-frequency steps: water slugs or flooding and the air-flow drift at 450 A (λ 1.76 → 1.53). The high-frequency part (above 20 Hz) stays clean everywhere, at 3–6 % deviation.

The CSV spectra look smooth because the bench tool smooths its results and the run was steady. The underlying cell behaviour is the same.

## 4. A plotting artefact made it look worse

The Nyquist line joined points across steps that the gates had rejected, so it drew long zig-zags. `plotting/nyquist.py` now leaves a gap where a step is missing (`_with_gaps`). The figures are `figures/spectra_recheck/nyquist_<cond>_gaps.png`.

## 5. Verdict per segment

Verdicts, using the median deviation of the points below 20 Hz from the smooth DRT model: **clean** below 3 %, **noisy < 20 Hz** 3–8 %, **unreliable < 20 Hz** 8 % or more.

| | clean | noisy < 20 Hz | unreliable < 20 Hz |
|---|---|---|---|
| 45 A | 67 | 0 | 0 |
| 60 A | 67 | 0 | 0 |
| 150 A | 33 | 34 | 0 |
| 450 A | 25 | 17 | 25 |

Every noisy or unreliable segment lies in the middle or outlet third of the air path.
For these segments:
- **R_ohmic**, **ReZ at 1 kHz** and **R_ct** are fine;
- **R_mt** and R_pol carry the LF scatter, about ±10 % at the 450 A outlet.

## 6. What to do

1. Re-run all four conditions on Databricks with run mode `rerun` so the window fix is applied.
2. In the thesis, quote R_mt at 150/450 A for the outlet segments with the larger uncertainty, or as "LF unreliable".
3. Future tests (see `test_plans/`):
   - a stable operating point before the sweep (constant air flow, not constant λ drift);
   - at least 3–5 cycles per LF step;
   - a repeat of the LF steps.
