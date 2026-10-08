# The bench tool's CSV spectra vs our FAMOS evaluation

Same measuring plate (R2-D2 gen1, 72 segments). Compared:

| | **CSV delivery** (`csv_files/Spectrum<n>_65degC_<I>A/`) | **FAMOS evaluation** (order 2612030) |
|---|---|---|
| What the files are | **Finished impedances**: one `z<n>.csv` per segment (mΩ·cm²), plus `z_cell.csv`, `hfr.csv` and `spectra.png`. The bench tool has already done the evaluation. | **Raw waveforms** (.DAT per card). Our bronze → silver → gold pipeline extracts the phasors, removes the chain and timing errors, and fits. |
| Order ID / Gamry / MF4 / Abgleich | none | all available |
| Conditions | 45, 60, 150, 449 A at 65 °C | 45, 60, 150, 450 A |
| Band | 0.2 Hz – 10.08 kHz, 48 points | 0.19 Hz – 3.8 kHz, about 41 points |
| Segments | all 72 | 67 measured (the rest are unmeasured or rebuilt) |
| Segment temperature (median) | 64.7 / 65.3 / 66.6 °C (45 / 150 / 450 A) | 58.2 / 61.0 / 63.9 °C |

## How the pipeline treats CSV data

1. **No chain-lag correction.** The tool's impedances are already corrected. Rotating them onto the plate median moved the HFR by up to 15 mΩ·cm².
2. **HFR = high-frequency intercept** (Z'' = 0). It agrees with the tool's `hfr_int` on every segment to 0.00 mΩ·cm². ReZ at 1 kHz agrees with the tool's `hfr_1kHz` to within 0.3 mΩ·cm².
3. **ECM and Kramers–Kronig on the capacitive band only.** This is the band between the HF and LF zero crossings of Z'', reaching up to about 2.5 kHz. Point 3 of the next section explains why.
4. The output format is the same as FAMOS: `gold/plate_summary.csv` in mΩ·cm², flow-coloured `nyquist.html` / `.png`, and 2D interpolated `plate_<param>_interp.png`. The tool's own values are carried in the columns `hfr_tool`, `hfr_1kHz_tool` and `hfr_300Hz_tool`.

## The differences that matter

### 1. The HF part above about 2 kHz is not the cell

Between 2 and 10 kHz every segment, at every current, turns inductive. Z' falls towards **0** at 10 kHz (minimum −0.1 mΩ·cm², median about 35). The whole-cell `z_cell.csv` reaches Re = 9 mΩ·cm² at 10 kHz.

A membrane cannot have zero resistance, so this loop belongs to the measuring chain. It is why:
- the plausibility checks flag *passivity* (Re Z ≤ 0 on 3 segments) and *segment frequency response* (phase spread 25° at 10 kHz);
- an ECM fitted to the full band put Rs at 0 (18 ± 21 mΩ·cm²). On the capacitive band Rs is 44–49 mΩ·cm², close to the intercept.

FAMOS stops at 3.8 kHz, so it never shows this loop. **Above about 2 kHz, only the FAMOS data describes the cell.**

### 2. The HFR is defined differently, but the numbers agree

| median, mΩ·cm² | 45 A | 150 A | 450 A |
|---|---|---|---|
| R_ohmic, FAMOS (top-band) | 52.3 | 48.2 | 51.1 |
| R_ohmic, CSV (intercept = tool hfr_int) | 52.6 | 47.1 | 48.1 |
| ReZ 1 kHz, FAMOS | 64.5 | 64.8 | 64.3 |
| ReZ 1 kHz, CSV | 62.6 | 53.4 | 55.4 |

The plate-level HFR matches to 0–3 mΩ·cm². At 150 and 450 A, ReZ at 1 kHz is about 10 mΩ·cm² lower in the CSV data. At 1 kHz the CSV spectra are already bending into the HF loop, so their 1 kHz value is not the same quantity as ours.

### 3. The spatial maps are mirror images left to right

At 450 A the profiles along the plate (x = 0 → 252 mm, in 63 mm bands) are:

| | x 0–63 | 63–126 | 126–189 | 189–252 |
|---|---|---|---|---|
| T, FAMOS [°C] | 64.7 | 63.9 | 62.7 | 62.1 |
| T, CSV [°C] | 65.3 | 66.6 | 68.2 | 69.3 |
| R_pol, FAMOS [mΩ·cm²] | 200 | 124 | 101 | 83 |
| R_pol, CSV [mΩ·cm²] | 78 | 88 | 129 | 204 |

Segment-by-segment correlation with FAMOS, as delivered and after mirroring the CSV in x (x → 252 − x):

| r | R_ohmic | R_pol | R_mt | T |
|---|---|---|---|---|
| 450 A as delivered | +0.26 | −0.78 | −0.74 | −1.00 |
| 450 A mirrored | +0.53 | **+0.97** | **+0.95** | +1.00 |
| 150 A as delivered | +0.43 | −0.55 | +0.26 | +0.12 |
| 150 A mirrored | +0.33 | **+0.79** | −0.14 | +0.11 |

Mirrored, the two measurements describe the same plate. At 450 A, FAMOS has the large arcs (mass transport) at the **air outlet**. It also has the hottest segment at the **coolant outlet**, which is physical: the coolant warms as it crosses the plate. As delivered, the CSV has both at the inlets.

**Two explanations fit:**
- the tool's x axis, or its channel-to-segment map, runs from the other end of the plate; or
- that test ran with the flows in the opposite direction.

The data cannot decide between them; the bench setup can. The pipeline therefore evaluates the files **as delivered** by default. Set `CSV_MIRROR_X = True` in the runner's settings cell (or `csv_mirror_x=True` in Config) to evaluate them mirrored. A mirrored run is cached separately.

### 4. Other differences

- **Temperature:** the CSV runs were 1–7 K warmer (65 °C setpoint). That lowers R_ct and changes humidification, so arc sizes are not directly comparable.
- **R_ct / R_mt split:** the CSV ECM sees the band only up to about 2.5 kHz, so the HF (charge-transfer) arc is less well defined. The R_ct spread is 2–7× larger than FAMOS (for example, 150 A: sd 56 vs 8 mΩ·cm²). R_pol, the sum of the two, is robust: 45 A 380 vs 353, 150 A 182 vs 210, 450 A 100 vs 119 mΩ·cm².
- **Kramers–Kronig:** few CSV segments pass at 2 %. The median of the maximum residual is 3.5–4.5 %, even on the capacitive band. Part of this is the 2 % σ the pipeline has to assume, because the tool gives no uncertainty per point.
- **Coverage:** CSV covers all 72 segments. FAMOS measures 67; segments 33 and 65–68 appear only on the CSV maps.
