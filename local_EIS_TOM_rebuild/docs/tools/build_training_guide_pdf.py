#!/usr/bin/env python3
"""
build_training_guide_pdf.py
===========================
Renders docs/TRAINING_GUIDE.md (+ docs/TRAINING_GUIDE_FILE_INDEX.md as
Appendix A) into docs/Local_EIS_Training_Guide.pdf: a cover page, a contents
list with page numbers, coloured headings and tables, PDF bookmarks.

    python docs/tools/make_file_index.py              # refresh the appendix first
    python docs/tools/build_training_guide_pdf.py     # from the package root

Needs, on the machine that builds it (not on Databricks):
    pandoc        Markdown -> HTML
    pypdf         reads the page of every heading for the contents list
    Chrome / Chromium (headless) to print; set $CHROME if it is not found

Two passes: the first PDF tells on which page every heading landed (its
link destinations), the second prints those numbers into the contents list.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
GUIDE = DOCS / "TRAINING_GUIDE.md"
INDEX = DOCS / "TRAINING_GUIDE_FILE_INDEX.md"
OUT = DOCS / "Local_EIS_Training_Guide.pdf"

TITLE = "Local EIS Pipeline"
SUBTITLE = "Training guide"
RUNNING = "Local EIS pipeline — Training guide"

CSS = r"""
@page { size: A4; margin: 20mm 17mm 18mm 17mm;
  @top-left { content: "%(running)s"; font: 7.5pt 'DejaVu Sans', sans-serif; color: #6b7280;
              border-bottom: 0.4pt solid #d1d5db; vertical-align: bottom; padding-bottom: 2mm; }
  @bottom-right { content: counter(page); font: 7.5pt 'DejaVu Sans', sans-serif; color: #6b7280; } }
@page cover { margin: 0; @top-left { content: none; border: none; } @bottom-right { content: none; } }
html { font-family: 'DejaVu Sans', 'Liberation Sans', sans-serif; font-size: 9.4pt; color: #1f2937; line-height: 1.42; }
body { margin: 0; }
a { color: #1f5c99; text-decoration: none; }
.cover { page: cover; height: 297mm; box-sizing: border-box; padding: 78mm 22mm 0 22mm; break-after: page; position: relative; }
.cover h1 { font-size: 34pt; color: #1f5c99; margin: 0; border: none; padding: 0; break-before: avoid; }
.cover .sub { font-size: 17pt; color: #1f2937; margin: 4mm 0 8mm; }
.cover .lead { font-size: 11pt; color: #6b7280; max-width: 150mm; line-height: 1.5; }
.cover table { position: absolute; left: 20mm; right: 20mm; top: 175mm; width: auto; font-size: 8.2pt; }
.toc { break-after: page; }
.toc h1 { break-before: avoid; }
.toc ol { list-style: none; margin: 0; padding: 0; }
.toc li { display: flex; align-items: baseline; margin: 0; }
.toc li a { color: inherit; }
.toc li .dots { flex: 1; border-bottom: 1.2pt dotted #9ca3af; margin: 0 1.5mm; transform: translateY(-2.5pt); }
.toc li.l1 { font-weight: 700; font-size: 11pt; color: #1f5c99; margin-top: 3.2mm; }
.toc li.l2 { font-weight: 700; font-size: 9.6pt; margin-top: 1.2mm; padding-left: 3mm; }
.toc li.l3 { font-size: 8.2pt; color: #4b5563; padding-left: 9mm; }
.toc li .pg { min-width: 7mm; text-align: right; }
h1 { font-size: 22pt; color: #1f5c99; margin: 0 0 4mm; break-before: page; }
h2 { font-size: 15.5pt; color: #1f5c99; margin: 7mm 0 2.5mm; break-after: avoid; }
h3 { font-size: 11.5pt; color: #1f2937; margin: 5mm 0 2mm; break-after: avoid; }
h4 { font-size: 10pt; color: #1f5c99; margin: 4mm 0 1.5mm; break-after: avoid; }
.appendix h3 { color: #1f5c99; font-size: 10.5pt; }
p { margin: 1.6mm 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.2pt; background: #f3f4f6; padding: 0 0.6mm; border-radius: 1mm; }
pre { background: #f3f4f6; border-left: 1.2mm solid #1f5c99; padding: 2.5mm 3mm; font-size: 7.4pt; line-height: 1.35;
      white-space: pre-wrap; word-break: break-word; break-inside: avoid; margin: 2.5mm 0; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { margin: 3mm 0; padding: 2.2mm 3.5mm; background: #e8f0f9; border-left: 1.2mm solid #1f5c99; break-inside: avoid; }
blockquote p { margin: 0.8mm 0; }
table { border-collapse: collapse; width: 100%%; margin: 2.5mm 0 3.5mm; font-size: 8pt; line-height: 1.3; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: #1f5c99; color: #fff; text-align: left; padding: 1.4mm 2mm; font-weight: 700; }
td { padding: 1.2mm 2mm; border-bottom: 0.4pt solid #e5e7eb; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f8fafc; }
td code, th code { font-size: 7.4pt; word-break: normal; overflow-wrap: break-word; }
th code { background: rgba(255,255,255,0.18); color: #fff; }
hr { display: none; }
ul, ol { margin: 1.5mm 0; padding-left: 6mm; }
li { margin: 0.6mm 0; }
.flow { display: grid; grid-template-columns: 52mm 1fr 40mm; column-gap: 5mm; margin: 4mm 0 5mm; font-size: 8pt; break-inside: avoid; }
.flow .hd { font-weight: 700; color: #4b5563; font-size: 7.6pt; letter-spacing: 0.3pt; margin-bottom: 2mm; }
.flow .in { background: #f3f4f6; border: 0.5pt solid #d1d5db; border-radius: 1.5mm; padding: 1.6mm 2.4mm; margin-bottom: 2mm; }
.flow .box { border-radius: 2mm; padding: 2.2mm 3.2mm; color: #fff; margin-bottom: 4.5mm; position: relative; }
.flow .box b { font-size: 10pt; display: block; margin-bottom: 0.6mm; }
.flow .box span { font-size: 7.6pt; opacity: 0.95; }
.flow .box:not(:last-child)::after { content: "▼"; position: absolute; left: 50%%; bottom: -4.4mm; color: #6b7280; font-size: 7pt; }
.flow .out { height: 15.7mm; display: flex; align-items: center; font-family: 'DejaVu Sans Mono', monospace; font-size: 7.6pt; color: #1f2937; margin-bottom: 4.5mm; }
.b1 { background: #b07a3b; } .b2 { background: #8a939c; } .b3 { background: #c9a227; } .b4 { background: #1f5c99; } .b5 { background: #1f2937; }
"""

FLOW = """
<div class="flow">
 <div><div class="hd">INPUTS</div>
  <div class="in">FAMOS .DAT cards<br>(datago upload / rows, Volume)</div>
  <div class="in">curr.csv · temp.csv<br>(EIS_config/&lt;plate&gt;/)</div>
  <div class="in">chain_gain.csv · Abgleich</div>
  <div class="in">Gamry .dta sweeps (datago, Volume)</div>
  <div class="in">bench .mf4 log</div>
  <div class="in">or CSV measurements (csv_files/)</div></div>
 <div><div class="hd">PROCESSING &nbsp;(pipeline/main.py: run_pipeline)</div>
  <div class="box b1"><b>BRONZE — pipeline/bronze.py</b><span>find steps, align cards, Gamry clock, fit phasors, Z = K·A_U/A_seg (no physics)</span></div>
  <div class="box b2"><b>SILVER — pipeline/silver.py</b><span>skew, channel lag, gates + ledger, DRT + circuit, R_ohmic, KK, tiers A/B/C</span></div>
  <div class="box b3"><b>GOLD — pipeline/gold.py</b><span>parameters, faults, 36-tile statistics, plate maps</span></div>
  <div class="box b4"><b>CHECKS — checks/*.py</b><span>Gamry vs aggregate, chain response, plausibility verdicts</span></div>
  <div class="box b5"><b>RUNNER — runner/ + notebooks</b><span>save folder, ECM fits, Delta tables, drop-down viewers</span></div></div>
 <div><div class="hd">OUTPUTS</div>
  <div class="out">→ bronze/</div><div class="out">→ silver/</div><div class="out">→ gold/</div>
  <div class="out">→ plausibility.csv,<br>&nbsp;&nbsp;gamry_comparison*</div>
  <div class="out">→ EIS_Results/…,<br>&nbsp;&nbsp;local_eis_* tables</div></div>
</div>
"""


def pandoc(md: str, id_prefix: str = "") -> str:
    cmd = ["pandoc", "-f", "gfm", "-t", "html5", "--wrap=none"]
    if id_prefix:
        cmd.append(f"--id-prefix={id_prefix}")
    return subprocess.run(cmd, input=md, capture_output=True, text=True,
                          check=True).stdout


def split_guide(md: str):
    """(cover facts table html, body markdown) -- the title block and the
    hand-written contents list are replaced by the cover and the generated
    contents."""
    head, _, rest = md.partition("\n## Contents\n")
    body = rest.split("\n# Part 1", 1)[1]
    body = "# Part 1" + body
    m = re.search(r"(\|\s*\|\s*\|\n\|---\|---\|\n(?:\|.*\|\n)+)", head)
    facts = pandoc(m.group(1)) if m else ""
    lead = re.search(r"\n\n(This is the training guide.*?)\n", head, re.S)
    return facts, (lead.group(1) if lead else ""), body


def headings(body_html: str, levels=(1, 2, 3)):
    out = []
    for m in re.finditer(r"<h([1-4])([^>]*)>(.*?)</h\1>", body_html, re.S):
        lvl = int(m.group(1))
        idm = re.search(r'id="([^"]+)"', m.group(2))
        if lvl in levels and idm:
            text = re.sub(r"<[^>]+>", "", m.group(3)).strip()
            out.append((lvl, idm.group(1), html.unescape(text)))
    return out


def toc_html(entries, pages=None) -> str:
    rows = []
    for lvl, hid, text in entries:
        pg = (pages or {}).get(hid, "")
        rows.append(f'<li class="l{lvl}"><a href="#{hid}">{html.escape(text)}</a>'
                    f'<span class="dots"></span><span class="pg">{pg}</span></li>')
    return ('<div class="toc"><h1 id="contents">Contents</h1><ol>'
            + "".join(rows) + "</ol></div>")


def chrome() -> str:
    for c in [os.environ.get("CHROME", ""), "chromium", "chromium-browser",
              "google-chrome", "chrome"]:
        if c and shutil.which(c):
            return shutil.which(c)
    hits = sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux*/chrome"))
    if hits:
        return str(hits[-1])
    sys.exit("no Chrome/Chromium found: set $CHROME")


def print_pdf(page_html: str, pdf: Path, workdir: Path) -> None:
    src = workdir / "guide.html"
    src.write_text(page_html, encoding="utf-8")
    subprocess.run([chrome(), "--headless", "--disable-gpu", "--no-sandbox",
                    "--no-pdf-header-footer", "--generate-pdf-document-outline",
                    f"--print-to-pdf={pdf}", src.as_uri()],
                   check=True, capture_output=True)


def heading_pages(pdf: Path) -> dict:
    """{heading id: page number} from the link destinations of the PDF."""
    from pypdf import PdfReader
    r = PdfReader(str(pdf))
    out = {}
    for name, dest in r.named_destinations.items():
        try:
            out[str(name).lstrip("/")] = r.get_destination_page_number(dest) + 1
        except Exception:                                    # noqa: BLE001
            pass
    return out


def build() -> Path:
    facts, lead, body_md = split_guide(GUIDE.read_text(encoding="utf-8"))
    body = pandoc(body_md)
    body = re.sub(r"<pre><code>[^<]*PROCESSING  \(pipeline/main.py: run_pipeline\).*?</code></pre>",
                  FLOW, body, count=1, flags=re.S)
    appendix = ""
    if INDEX.is_file():
        idx = INDEX.read_text(encoding="utf-8")
        idx = re.sub(r"^# .*$", "# Appendix A — File reference (generated)", idx,
                     count=1, flags=re.M)
        appendix = '<div class="appendix">' + pandoc(idx, "fi-") + "</div>"
    entries = headings(body) + [(lvl, hid, t) for lvl, hid, t in
                                headings(appendix, (1, 2))]
    cover = (f'<div class="cover"><h1>{TITLE}</h1><div class="sub">{SUBTITLE}</div>'
             f'<div class="lead">{html.escape(lead)}</div>{facts}</div>')

    def page(pages):
        return ("<!doctype html><html><head><meta charset='utf-8'><title>"
                f"{TITLE} — {SUBTITLE}</title><style>"
                + CSS % {"running": RUNNING} + "</style></head><body>"
                + cover + toc_html(entries, pages) + body + appendix
                + "</body></html>")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        first = td / "pass1.pdf"
        print_pdf(page({hid: "000" for _, hid, _ in entries}), first, td)
        pages = heading_pages(first)
        missing = [hid for _, hid, _ in entries if hid not in pages]
        if missing:
            print(f"  {len(missing)} headings without a page (e.g. {missing[:3]})")
        print_pdf(page(pages), OUT, td)
    return OUT


if __name__ == "__main__":
    p = build()
    print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
