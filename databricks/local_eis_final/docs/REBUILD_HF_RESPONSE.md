# Rebuilding the high-frequency response for all conditions

The frozen high-frequency response (`EIS_Results/<order>/hf_response/hf_response.csv`)
corrects each channel above 1 kHz: a phase line per segment and a magnitude per
card. It is measured on the corrected spectra, so it belongs to the settings it
was built with. After changing any of these, build it again from scratch:

`SKEW_MODEL`, `CHANNEL_LAG`, the chain gain file, `CARD_GAIN`, the Gamry card phase.

All conditions of an order must then be evaluated with the **one** new file —
otherwise they are corrected differently above 1 kHz and R_ohmic cannot be
compared between them (on 2612030 / 450 A a different file moved R_ohmic of
the Karte_2 segments by up to 11 %).

## Steps (runner notebook)

| # | Where | Set / do | What happens |
|---|---|---|---|
| 0 | widgets | select **all** conditions (45, 60, 150, 450 A); `cache_write = yes` | the file can only be built from the selected conditions; with cache writing off it is written to /tmp, where `'auto'` does not look |
| 1 | settings cell | `SKEW_MODEL = 'none'` (default now) and `HF_RESPONSE = ''` | the runs are made **without** any response file, so the new one is not composed on top of the old |
| 2 | Run all | pipeline cell evaluates every condition (new cache key: the settings changed) | |
| 3 | "High-frequency response across conditions" cell | runs automatically in Run all (no file is active, so it builds) | the old file is copied aside as `hf_response_<date-time>.csv`; the new one is written; the leave-one-out table must show phase sd ≤ 5° and card step ≤ 5 % for every held-out current; the cell prints "built from scratch" |
| 4 | settings cell | `HF_RESPONSE = 'auto'` (keep `HF_RESPONSE_REBUILD = False`) | |
| 5 | Run all | every condition is re-evaluated with the new file (its content is in the cache key) | the HF cell now says "kept as it is (frozen)" |
| 6 | HF cell output | the table "Which response file did each condition's result use?" | every condition must show the same file and no "older than the current file" flag |

After step 6 the file is frozen: later "Run all" sessions load every condition
from the cache and do not rebuild it. Rebuild only on purpose (repeat 1–6).

## Checks after the rebuild

* `plausibility.csv` of each condition: "segment frequency response (in-situ)"
  phase spread ≤ 5° and card steps ≤ 5 %.
* R_ohmic per segment across 45 / 60 / 150 / 450 A: with one file for all,
  differences between conditions are the cell's, not the correction's.
* `silver/skew_model.csv` shows `disabled` on every card (SKEW_MODEL = 'none').

## What the code now guards

* `hf_response.build()` refuses runs that differ in `skew_model`, `channel_lag`,
  `channel_lag_model`, `gamry_card_phase`, `card_hf_phase`, `card_gain` or
  `gain_file`, and runs evaluated with different response files.
* The previous file is never lost: it is copied to `hf_response_<date-time>.csv`
  before the new one is written.
* The HF cell says whether the new file was built from scratch or on top of an
  older one, and lists which file each condition's result used.
