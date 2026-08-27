"""The identifier must name what it can and refuse what it cannot."""

from __future__ import annotations

import zipfile

import identify_file as I


def test_a_famos_file_is_recognised_as_already_supported(tmp_path):
    f = tmp_path / "Karte_1.DAT"
    f.write_bytes(b"|CF,2,1,1;|CK,1,3,1,1;" + b"\x00" * 200)
    text = "\n".join(I.identify(f))
    assert "imc FAMOS raw" in text
    assert "--dat" in text, "must say it is already supported, not just named"


def test_an_ambiguous_extension_is_called_out(tmp_path):
    """.ddf is claimed by unrelated products, so the name settles nothing."""
    f = tmp_path / "bench.ddf"
    f.write_bytes(bytes(range(256)))
    text = "\n".join(I.identify(f))
    assert "at least three unrelated products" in text
    assert "BINARY, no known signature" in text


def test_a_container_is_opened_rather_than_merely_named(tmp_path):
    """"It is a zip" is not an answer; the entry names are."""
    f = tmp_path / "bench.ddf"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("data.xml", "<root/>")
        z.writestr("channels.bin", b"\x00" * 8)
    text = "\n".join(I.identify(f))
    assert "zip container" in text
    assert "data.xml" in text and "channels.bin" in text


def test_a_text_file_reports_its_delimiter(tmp_path):
    f = tmp_path / "p1.csv"
    f.write_text("timestamp\ts1\ts2\ts3\ntimeshifts\t0\t1.1\t2.2\n")
    text = "\n".join(I.identify(f))
    assert "TEXT" in text
    assert "3 '\\t' separator(s)" in text
    assert "timeshifts" in text, "the first lines must be shown verbatim"


def test_a_truncated_zip_does_not_crash_the_report(tmp_path):
    """A half-copied file off a share is a normal thing to be handed."""
    f = tmp_path / "broken.ddf"
    f.write_bytes(b"PK\x03\x04" + b"\x00" * 40)
    text = "\n".join(I.identify(f))
    assert "zip listing failed" in text
    assert "first bytes:" in text, "the hex must still be reported"


def test_a_directory_says_so_and_lists_it(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    text = "\n".join(I.identify(tmp_path))
    assert "is a DIRECTORY" in text and "a.csv" in text


def test_a_missing_file_is_reported_not_raised(tmp_path):
    assert "does not exist" in "\n".join(I.identify(tmp_path / "nope.ddf"))
