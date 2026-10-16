# gen2 (blue / Naboo) plate files go here

Put the **Naboo** calibration delivery in this folder, then upload the whole
`EIS_config/` folder to the Volume (see `SETUP_SHARED_FILES.md`):

```
EIS_config/gen2/
    curr.csv          REQUIRED  Naboo coefficients/curr.csv (72 rows c0;c1)
    temp.csv          REQUIRED  Naboo coefficients/temp.csv (4 rows c0;c1)
    chain_gain.csv    optional  written by the runner's "Chain response" cell
    Abgleich/         optional  the Naboo delivery: coefficients/, bode/, Step*_<T>Grad.csv
```

Without `curr.csv` and `temp.csv` a gen2 run stops with a message naming this
folder. It never falls back to the gen1 (Kashyyyk) files: the two
calibrations differ by ~1.45x in absolute level, so borrowing them would
scale every impedance wrongly without any error.

Note for the chain response: the Naboo bode index does not match its
`curr.csv` row order (cross-check r = +0.41). The runner detects that and
writes the index-free plate median instead (`shared` mode), which still
removes the common −11° roll-off at 4.5 kHz.
