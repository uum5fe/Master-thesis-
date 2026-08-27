"""The DDF probe: does it find a layout it was not told, and refuse a guess?

The failure mode that matters is not "cannot read the file" -- that is loud
and harmless. It is reading it at the wrong alignment and returning numbers
that look like a measurement and are not. So every test here checks the
refusals as hard as it checks the successes.
"""

from __future__ import annotations

import numpy as np
import pytest

import ddf_source as D


FS = 10_000.0


def plant(path, n=20000, ch=16, dtype="<f4", header=True, pad_to=1024):
    """A DASYLab-shaped file with a KNOWN layout, for the probe to rediscover."""
    head = b""
    if header:
        head = (f"DASYLab V13.00.00\r\n"
                f"Number of channels\t{ch}\r\n"
                f"Sample rate\t{FS}\r\n"
                f"Block size\t1024\r\n").encode("latin-1")
        if pad_to:
            head += b"\x00" * ((-len(head)) % pad_to)

    t = np.arange(n) / FS
    sig = np.zeros((n, ch))
    for c in range(ch):
        sig[:, c] = (0.78 + 0.004 * np.sin(2 * np.pi * 298.3 * t + c * 0.2)
                     + 5e-4 * np.random.default_rng(c).standard_normal(n))
    path.write_bytes(head + sig.astype(dtype).tobytes())
    return len(head)


def test_the_probe_recovers_a_layout_it_was_never_told(tmp_path):
    f = tmp_path / "m.ddf"
    offset = plant(f)
    p = D.probe(f)

    best = p.candidates[0]
    assert best.n_channels == 16
    assert best.dtype_name == "float32-le"
    assert best.offset == offset
    assert best.corroborated, "an answer resting on smoothness alone is not one"


def test_the_recovered_layout_reads_back_the_exact_samples(tmp_path):
    """Bit-exact, not approximately: a byte offset is right or it is not."""
    f = tmp_path / "m.ddf"
    offset = plant(f, n=4096, ch=8)
    p = D.probe(f)
    best = p.candidates[0]

    got = D.read_ddf(f, best.offset, best.n_channels, best.dtype_name)
    raw = np.frombuffer(f.read_bytes()[offset:], dtype="<f4").reshape(-1, 8)
    assert got.shape == raw.shape
    assert np.array_equal(got, raw.astype(np.float64))


def test_the_interleave_period_is_measured_not_guessed(tmp_path):
    """Autocorrelation must find the channel count with the header removed.

    This is the part that does not depend on DASYLab writing a helpful
    header, so it has to stand on its own.
    """
    for ch in (8, 16, 24):
        f = tmp_path / f"h{ch}.ddf"
        plant(f, n=20000, ch=ch, header=False)
        stride, strength = D.stride_from_autocorrelation(f.read_bytes(), 0)
        assert stride == ch, f"measured {stride} for {ch} channels"
        assert strength > 0


def test_zero_padding_does_not_win_by_being_smooth(tmp_path):
    """A run of exact zeros is maximally smooth and is not data.

    Without the zero penalty and the frame-equivalence collapse, the search
    settles a few hundred bytes short of the real offset and reports an
    excellent score for reading the padding.
    """
    f = tmp_path / "m.ddf"
    offset = plant(f, pad_to=4096)
    p = D.probe(f)
    best = p.candidates[0]
    assert best.offset == offset
    assert best.span[0] > 0.5, ("the winning window must start on signal, not "
                                "on padding")


def test_random_bytes_are_refused_rather_than_interpreted(tmp_path):
    """The whole point. Noise must not produce a confident layout."""
    f = tmp_path / "noise.ddf"
    f.write_bytes(np.random.default_rng(0).bytes(400_000))
    report = D.probe(f).report()
    assert "VERDICT:" in report
    assert "read it with" not in report, "noise must not get a read command"


def test_an_uncorroborated_leader_is_not_reported_as_the_answer(tmp_path):
    """The decision rule itself, not a fixture that happens to trigger it.

    Smoothness alone cannot tell N similar channels read as 1 from the truth,
    so a leader resting on it must not get a read command however well it
    scored. Asserted against a constructed Probe because building a file that
    reliably defeats BOTH the header and the autocorrelation is harder than
    the rule is -- and it is the rule that has to hold.
    """
    high = D.Layout(offset=0, n_channels=1, dtype_name="float32-le",
                    score=0.99, finite_fraction=1.0, smoothness=0.01,
                    span=(0.7, 0.8), n_samples=4096, corroborated="")
    report = D.Probe(path=tmp_path / "x.ddf", size=1, candidates=[high]).report()

    assert "export route" in report
    assert "read it with" not in report, (
        "an uncorroborated layout must never be handed to the user as a "
        "command to run")


def test_a_corroborated_leader_does_get_a_read_command(tmp_path):
    """The other half of the same rule, so it is not vacuously satisfied."""
    good = D.Layout(offset=1024, n_channels=16, dtype_name="float32-le",
                    score=1.3, finite_fraction=1.0, smoothness=0.2,
                    span=(0.774, 0.786), n_samples=4096,
                    corroborated="header+autocorrelation+aligned")
    report = D.Probe(path=tmp_path / "x.ddf", size=1, candidates=[good]).report()

    assert "read it with" in report
    assert "--offset 1024" in report and "--channels 16" in report


def test_the_verdict_always_shows_the_range_to_sanity_check(tmp_path):
    """A number a human can reject at a glance beats a score they cannot."""
    f = tmp_path / "m.ddf"
    plant(f)
    report = D.probe(f).report()
    assert "value range" in report
    assert "CHECK THIS" in report


def test_to_csv_writes_something_the_csv_path_can_read(tmp_path):
    f = tmp_path / "m.ddf"
    offset = plant(f, n=512, ch=4)
    out = D.to_csv(f, tmp_path / "out.tsv", offset, 4)

    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == ["s1", "s2", "s3", "s4"]
    assert len(lines) == 513
    assert all(len(l.split("\t")) == 4 for l in lines[1:])
