# R2-D2 logger sample

2500 data rows from a real point file, taken from **inside the excitation
burst**, kept as a format fixture so the reader and the phase handling are
tested against the hardware's own output and not only against synthetic data.
The full file is 19 417 rows (1.765 s).

    metadata.csv   the sidecar, verbatim
    p1.csv         the two header rows + rows 4001..6500 of the original
                   (t = 0.364 .. 0.591 s, ~210 cycles of the tone)

Rows 1..2750 and 14840..19417 of the original are lead-in and lead-out with no
excitation, which is why the excerpt starts where it does — and why the
pipeline windows to the burst rather than fitting the whole record.

Properties of this recording, all measured rather than assumed — see
`docs/GEN2_PLATE_AND_CSV_PIPELINE.md` §5:

    80 channels          s1..s72, uc1..uc4, temp1..temp4
    fs                   11 001.10 Hz  (90.90 us per row)
    channel scan         1.1 us apart, spanning 86.898 us = 96 % of a row
    s columns            current density A/cm2 (the logger applied
                         coefficient set "Coruscant"); j = 1.37..2.46
    temp columns         degC, 65.4 .. 72.4
    uc1..uc4             two differential pairs around a 0.61 V cell,
                         disagreeing by 28 % in ac amplitude
    excitation           a burst, ~0.25 s to ~1.35 s in the full file,
                         404x the noise floor in this excerpt
    tone in the record   923.09 Hz -- which is an ALIAS: the channel scan
                         measures the analogue phase ramp and it corresponds
                         to 10 078 Hz, i.e. this point sits above the
                         5 500 Hz Nyquist frequency
    persistent artefact  ~999 Hz at ~6x the floor, present in the silent
                         stretches; it is what a naive peak-pick on a
                         truncated file locks onto

The full sweep is a folder of these, one per frequency, plus one metadata.csv.
