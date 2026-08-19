# Calibration campaigns

A campaign ("Abgleichdaten") is a folder of `Step<n>_<T>Grad.csv` files plus a
`coefficients/` folder. It belongs to a **plate**, not to a measurement order,
so the viewer identifies it by folder name and never asks for an order id.

```
<any root>/Kashyyyk/
    Step1_20Grad.csv  Step2_40Grad.csv  …  Step6_20Grad.csv
    coefficients/curr.csv     72 lines of c0;c1
    coefficients/temp.csv      4 lines of c0;c1
    bode/*.DTA                 Gamry sweeps; not read by the viewer
```

Point `EIS_CALIBRATION_ROOT` at a folder containing campaigns, or leave it
unset — the FAMOS and results roots are searched too, so a campaign stored
beside the measurements is found without extra configuration. Then set **File
format → Calibration sweeps** and pick the campaign.

## What the file format means

One line per segment:

```
s1:<TAB>temp1=3.000357V;temp2=2.464455V;temp3=2.484830V;temp4=3.215963V;
        <TAB>i_s=-0.000004A;u_s=0.006492V;<TAB>i_s=0.250458A;u_s=0.316003V;…
```

the four plate temperature-sensor voltages, then a current sweep — five points
from 0 to 1 A — of injected current against measured shunt voltage.

**`Grad` is the plate temperature in °C, not an angle.** Putting the sensor
voltages through `temp.csv` (`T = (V − c0) / c1`) reproduces the file's label,
which is how that was settled rather than assumed.

## How the shipped coefficients relate to the sweeps

`coefficients/curr.csv` is reproduced from the raw data by fitting each
segment's sweep **through the origin** — `u_s = H·i_s`, no intercept — and then
fitting that sensitivity against temperature, `H(T) = c0 + 1e-3·c1·T`.

That convention is not a guess. Fitting through the origin makes the ratio
between the reconstruction and the shipped coefficient identical for all 72
segments to within 1e-4; allowing an intercept leaves a scatter a hundred times
larger. On the three campaigns supplied:

| Campaign | Steps | Range | Constant | Spread over 72 segments |
| --- | --- | --- | --- | --- |
| Kashyyyk | 6 | 20 … 90 °C | 2.9510 | 3.3 × 10⁻⁵ |
| Kashyyyk_mod | 6 | 22 … 92 °C | 3.0579 | 1.3 × 10⁻⁴ |
| Naboo | 12 | −38 … 92 °C | 3.0607 | 4.7 × 10⁻⁵ |

**What that constant is cannot be derived from these files.** It is a property
of the measurement chain, and it changed between Kashyyyk and the later two.
The viewer therefore reports it rather than assuming it — and the useful part
is not its value but its *constancy*: a segment that departs from the plate's
own constant was calibrated from something other than these sweeps, and the
`ratio_c0_dev_pct` map finds it.

## What the tab evaluates

| | Question it answers |
| --- | --- |
| **Worst linearity** | Do the five sweep points lie on a line through the origin? A segment that does not has a shunt or amplifier fault, and no coefficient will rescue it. |
| **Sensitivity map** `H_at_60C`, `H_c0` | How much shunt voltage each segment gives per amp. |
| **Temperature coefficient** `H_c1`, `drift_pct_over_span` | How that moves with temperature. A segment far from the others has a different thermal path, not a different gain. |
| **Departure from the chain constant** `ratio_c0_dev_pct` | Whether the shipped coefficient belongs to these sweeps. |
| **Repeat drift** | A campaign that starts and ends at the same nominal temperature is asking whether the plate came back. The answer is per segment. |
| **Step table** | What each file claims against what the sensors read. |

Clicking a segment on the map shows its raw sweeps, one line per temperature,
and its sensitivity against temperature with the fitted line — so a suspect
number on the map can be traced to the measurement behind it.

## Something the data shows

The Naboo campaign's sensors read about **+1.7 °C above the step labels**;
Kashyyyk's reproduce them to 0.02 °C. Either the chamber setpoint and the plate
genuinely differ, or `temp.csv` carries an offset. The viewer says so on the
Calibration tab rather than quietly using one or the other — the labels are
only used when `temp.csv` is missing.

## What is not read

`bode/*.DTA` are Gamry sweeps of the chain's frequency response (100 kHz → 1 Hz
at 500 mA). They are the input to the pipeline's optional `--gain` correction,
not to the calibration evaluation, and the viewer does not read them yet.

---

## Appendix: reading a plate drawing

`tools/extract_label_pads.py` reads the segment numbers off a coordinates PDF
and reports the pad each one sits on, calibrated from the axis tick labels
whose millimetre values are printed on the drawing:

```bash
python tools/extract_label_pads.py Coordinates.pdf
python tools/extract_label_pads.py green.pdf blue.pdf --compare
```

It reproduces the documented Gen-1 layout exactly from the green drawing, which
is why its readings on a new drawing can be trusted.

It gives label pads, **not** segment boundaries. A label pad marks where a
number is printed; on these boards the segments are formed by routing pads to
shunts, so the boundaries live in the netlist. Use this to see what changed
between generations, then take the pad-to-segment mapping from the board file.
