# famos_rename — put your own channel names on a FAMOS file

A standalone tool. It imports nothing from the EIS pipeline and the pipeline
imports nothing from it: copy `famos_rename.py` anywhere and it works. Only
Python 3.10+ and NumPy — and NumPy only for reading samples, not for renaming.

## The problem

DASYLab names the channels it records after their position on the card, not
after the thing wired to that position. A card carrying segments 64–79 is
written out as:

```
"0", "1", "2", ... "15"
```

The measurement is fine — only the labels are wrong. But a spectrum filed
under segment 3 that belongs to segment 67 does not look wrong at all, which
is why this is worth a tool rather than a mental note.

## Nothing is guessed

The new names come from you and only from you. This tool will not read a
range out of the file name, will not assume the channels are in ascending
order, and will not fill in a name it was not given.

## The workflow

```bash
# 1. what is in the file
python famos_rename.py list KANAL_6479.DAT

# 2. a CSV to edit -- new_name starts as the name already in the file
python famos_rename.py template KANAL_6479.DAT -o names.csv

# 3. edit the new_name column, then apply it
python famos_rename.py apply KANAL_6479.DAT --names names.csv --out renamed.DAT

# 4. renamed.DAT is an ordinary FAMOS file -- use it for the EIS evaluation
```

`names.csv` after step 2, with the third column yours to change:

```csv
channel,label_in_file,new_name
0,0,64
1,1,65
...
15,15,79
```

For a straightforward case, skip the template and say it inline — names are
given in header order:

```bash
python famos_rename.py apply FILE --names 64-79 --out renamed.DAT
python famos_rename.py apply FILE --names UC1,65-79 --out renamed.DAT
python famos_rename.py apply FILE --names UC1,64,65,temp1 --out renamed.DAT
```

`64-79` is only shorthand for typing the sixteen numbers out. Any name works:
`UC1`, `temp1`, `seg_67`, whatever your evaluation expects.

Add `--dry-run` to see the table it would write and write nothing.

## What it writes

A FAMOS file identical to the input except for the channel-name (`|CN`) keys.
Their length fields are rebuilt so the keys after them stay findable; every
other key and every single sample byte is copied through untouched, and the
input file is never modified. Any reader opens the result exactly as it opened
the original, with the right names on the channels.

`apply` re-reads the copy and compares the sample region against the source
before it reports success, so the copy is not taken on trust. To check again
later:

```bash
python famos_rename.py verify renamed.DAT --against KANAL_6479.DAT
```

## If your evaluation would rather have a CSV

The renamed `.DAT` is the deliverable; this is a convenience.

```bash
python famos_rename.py export renamed.DAT --out data.csv
python famos_rename.py export renamed.DAT --out data.csv --channels 64,70,79 --step 10
```

One row per sample, so the whole file is large — `--step N` keeps every Nth.

## Using it from Python

```python
import famos_rename as F

head = F.read_header("KANAL_6479.DAT")
print(head.names, head.n_channels, head.fs)

out = F.rename("KANAL_6479.DAT", "renamed.DAT",
               [str(s) for s in range(64, 80)])

seg67 = F.read_data(out, channels=["67"])[:, 0]   # memory-mapped
```

## What it refuses

Every one of these is a wrong-but-plausible result if it goes through, so all
of them stop before anything is written:

| refusal | why |
| --- | --- |
| a list that is not exactly one name per channel | pairing them off as far as they go renames some channels and leaves the rest quietly wrong |
| a duplicate name | the two channels become impossible to tell apart |
| a blank name | keep a name by writing it out, so a skipped row is never mistaken for a decision |
| a name a FAMOS header cannot hold | the header is 8-bit latin-1 |
| a frame whose `\|CP` keys disagree about its size | reading padding as a plain interleave puts every sample after the first in the wrong channel |
| `--out` naming the input | the original is the only copy of the measurement |
| a file with no `\|CN` keys | there is nothing to rename |

## Tests

```bash
python -m pytest test_famos_rename.py -q
```

32 tests, on synthetic FAMOS files with a known channel table. They weigh the
refusals as heavily as the renames, and hold `rename` to the standard that
matters for rewriting a measurement file: not one sample byte moves.
