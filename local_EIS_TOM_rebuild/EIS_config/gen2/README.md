# gen2 (blue / Naboo) plate files

The Naboo calibration delivery, ready to upload with the rest of
`EIS_config/` to the Volume (see `SETUP_SHARED_FILES.md`):

```
EIS_config/gen2/
    curr.csv          Naboo coefficients/curr.csv (72 rows c0;c1)
    temp.csv          Naboo coefficients/temp.csv (4 rows c0;c1)
    chain_gain.csv    built from Abgleich/bode, index-free plate median
    Abgleich/         the Naboo delivery: coefficients/, bode/ (Tatooine_*.DTA),
                      Step1..Step12_<T>Grad.csv (-40 .. 90 degC)
```

A gen2 run never falls back to the gen1 (Kashyyyk) files: the two
calibrations differ by ~1.45x in absolute level, so borrowing them would
scale every impedance wrongly without any error.

The chain response: the Naboo bode index does not match its `curr.csv` row
order (cross-check r = +0.41, ratio scatter 12 %), so `chain_gain.csv` is the
index-free plate median (`shared` mode, the same file the runner would build),
which still removes the common -11 deg roll-off at 4.5 kHz.

Checks on the delivery (readers/abgleich.verify): 12 temperature steps,
linearity r^2 >= 0.9997, TCR 0.404 .. 0.415 %/K, implied R/K constant to
0.6 % across segments (segment 37 flagged as the one outlier).
