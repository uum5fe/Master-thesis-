"""The log must survive a console that cannot spell.

A Windows console is cp1252. The banners use box drawing and the results are
full of micro, ohm, degree and plus-minus. Without care, logging catches the
UnicodeEncodeError and routes it to `handleError`, which prints a "--- Logging
error ---" traceback in place of every such line -- the run is unharmed, and
its output is unreadable.

These tests drive the pieces directly rather than through `sys.stdout`:
pytest replaces stdout for its own capture, so a test that monkeypatched it
would be testing pytest's plumbing instead of ours.
"""

from __future__ import annotations

import io
import logging

import pytest

import utils


class _LegacyConsole(io.TextIOWrapper):
    """cp1252, refuses to reconfigure, and has no byte buffer to borrow."""

    def reconfigure(self, **kw):
        raise OSError("cannot reconfigure this stream")

    @property
    def buffer(self):
        raise AttributeError("buffer")


def _console(encoding="cp1252", legacy=True):
    raw = io.BytesIO()
    cls = _LegacyConsole if legacy else io.TextIOWrapper
    return raw, cls(raw, encoding=encoding, errors="strict",
                    line_buffering=True)


def _logger_on(stream) -> logging.Logger:
    log = logging.getLogger(f"eis.test.{id(stream)}")
    log.handlers.clear()
    handler = utils._SafeStreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


def _text(raw, console) -> str:
    console.flush()
    return raw.getvalue().decode("cp1252", errors="replace")


# ---------------------------------------------------------------------------


def test_a_legacy_console_gets_ascii_rules():
    """Box drawing is not available, so the rules must not use it."""
    _raw, console = _console()
    heavy, light = utils._rules_for(console)
    assert (heavy, light) == ("=", "-")


def test_a_utf8_console_keeps_the_real_rules():
    """Nothing is degraded where the console can carry it."""
    _raw, console = _console(encoding="utf-8", legacy=False)
    heavy, light = utils._rules_for(console)
    assert (heavy, light) == ("═", "─")


def test_a_character_the_console_lacks_does_not_lose_the_line():
    """The line matters more than the symbol in it."""
    raw, console = _console()
    log = _logger_on(console)
    log.info("segment 33: R_ohmic 60 mΩ·cm² dropped")

    out = _text(raw, console)
    assert "Logging error" not in out
    assert "segment 33" in out
    assert "dropped" in out, "the end of the line survived the bad character"
    assert "Ω" not in out          # replaced, not raised


def test_an_encodable_character_is_left_alone():
    """cp1252 has micro and degree. Only what it lacks should be replaced."""
    raw, console = _console()
    log = _logger_on(console)
    log.info("scan offset +80.3 µs = +291°")

    out = _text(raw, console)
    assert "Logging error" not in out
    assert "+80.3 µs = +291°" in out


def test_the_stream_is_put_into_utf8_when_the_platform_allows(monkeypatch):
    """The real fix, where it is available: reconfigure and keep everything."""
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252", line_buffering=True)
    monkeypatch.setattr(utils.sys, "stdout", console)

    stream = utils._console_stream()
    assert stream.encoding.lower().replace("-", "") == "utf8"
    assert utils._rules_for(stream) == ("═", "─")


def test_banner_and_section_use_whatever_the_console_can_carry():
    """Both helpers must read the same resolved rule characters."""
    raw, console = _console()
    log = _logger_on(console)
    heavy, light = utils._rules_for(console)
    monkey = (utils._RULE, utils._LIGHT)
    utils._RULE, utils._LIGHT = heavy * 75, light
    try:
        utils.banner("import and geometry checks", log)
        utils.section("schedule detection (blind, consensus)", log)
    finally:
        utils._RULE, utils._LIGHT = monkey

    out = _text(raw, console)
    assert "Logging error" not in out
    assert "import and geometry checks" in out
    assert "schedule detection" in out
    assert "=" * 20 in out and "---" in out
