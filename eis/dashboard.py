"""Interactive plate view: every segment, every frequency, one HTML file.

What this exists to fix
-----------------------
A static plate map answers "where is the resistance high?" and nothing else.
The question that actually comes next - *why* is it high there - needs the
spectrum of that segment, and walking from a coloured square to the right
figure in a folder of eighty is where the analysis stops happening.

So the map and the spectra live on one page and are linked: click a segment,
see its Nyquist and Bode with error bars and its circuit fit; drag the
frequency slider and watch ``|Z|`` or the phase redraw across the plate, which
separates the losses that are spatially smooth and frequency-flat (ohmic) from
the ones that follow the flow field and only appear in a band (kinetic and
transport).

Every segment is drawn
----------------------
Including the bad ones.  A segment whose quality score is low is drawn
desaturated with a dashed outline, and one that produced no usable point at all
is drawn empty with its status written on it.  Nothing is missing from the
plate, so nothing has to be remembered as missing.

Self-contained by construction
------------------------------
One file, no server, no external libraries, no network: the data is embedded as
JSON and the drawing is plain SVG built by about two hundred lines of
JavaScript.  It can be mailed, committed, or opened from a stick five years
from now.  Frequency-resolved maps are computed in the browser from the spectra
that are already on the page rather than shipped one map per frequency, which
keeps a full 80-segment plate under a megabyte.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eis.pipeline.config import PipelineConfig
from eis.pipeline.gold import FREQUENCY_KEYS, ConditionResult, map_label, scalar_map

#: Keys computed in the browser from the embedded spectra.
_BROWSER_KEYS = sorted(FREQUENCY_KEYS)


def _round(values, digits: int = 6) -> list:
    array = np.asarray(values, float)
    array = np.where(np.isfinite(array), array, np.nan)
    return [None if not np.isfinite(v) else float(f"%.{digits}g" % v) for v in array]


def build_payload(
    results: dict[str, ConditionResult], cfg: PipelineConfig
) -> dict:
    """Everything the page needs, as plain JSON-serialisable data."""
    from eis.viz import default_segment_grid

    area = cfg.geometry.segment_area_cm2
    asr = area * 1e3

    all_segments = sorted({
        s for result in results.values() for s in result.segments
    })
    coords = cfg.geometry.segment_coords or default_segment_grid(
        all_segments, cfg.geometry.plate_w_cm, cfg.geometry.plate_h_cm
    )

    static_keys = [
        k for k in cfg.report.heatmap_parameters if k not in FREQUENCY_KEYS
    ]
    frequency_keys = [
        k for k in cfg.report.heatmap_parameters if k in FREQUENCY_KEYS
    ] or _BROWSER_KEYS

    conditions: dict[str, dict] = {}
    for name, result in results.items():
        maps = {
            key: {str(s): v for s, v in scalar_map(result, key, area).items()}
            for key in static_keys
        }
        segments: dict[str, dict] = {}
        for segment, record in sorted(result.segments.items()):
            spectrum = record.spectrum_all
            entry = {
                "card": record.card,
                "status": record.status,
                "quality": round(float(record.quality), 4),
                "active": bool(record.active),
                "flags": record.flags,
                "physical": bool(record.physical_units),
                "note": record.note,
                "coords": [round(float(c), 4) for c in coords[segment]]
                if segment in coords else None,
                "f": _round(spectrum.f),
                "zr": _round(spectrum.Z.real * asr),
                "zi": _round(spectrum.Z.imag * asr),
                "sig": _round(spectrum.sigma_rel),
                "coh": _round(spectrum.coherence),
                "used": [int(u) for u in spectrum.used_mask],
                "rs": None if not np.isfinite(record.hfr.rs_ohm)
                else round(float(record.hfr.rs_ohm * asr), 5),
                "kk": None if record.kk is None
                else round(float(record.kk.max_residual_pct), 4),
            }
            if record.ecm is not None and len(getattr(record.ecm, "Z_fit", [])):
                entry["fr"] = _round(record.ecm.Z_fit.real * asr)
                entry["fi"] = _round(record.ecm.Z_fit.imag * asr)
                entry["model"] = record.ecm.model
            segments[str(segment)] = entry

        conditions[name] = {
            "maps": maps,
            "segments": segments,
            "n_active": len(result.active_segments),
            "n_total": len(result.segments),
            "status_counts": result.status_counts(),
            "sync_passed": bool(result.sync.passed),
            "reference_card": result.sync.reference_card,
            "duration_s": round(float(result.duration_s), 2),
            "tone_check": result.tone_check,
            "common_mode": {
                str(card): {
                    "delay_ns": round(float(cm.delay_s * 1e9), 2),
                    "applied": bool(cm.applied),
                    "note": cm.note,
                }
                for card, cm in result.common_mode.items()
            },
        }

    first = next(iter(results.values()))
    return {
        "meta": {
            "measurement_id": first.measurement_id,
            "param_hash": first.provenance.get("param_hash", ""),
            "git_sha": first.provenance.get("git_sha", ""),
            "created_utc": first.provenance.get("created_utc", ""),
            "pipeline_version": first.provenance.get("pipeline_version", ""),
            "segment_area_cm2": area,
            "good_quality": cfg.quality.good_quality,
        },
        "plate": {
            "w": cfg.geometry.plate_w_cm,
            "h": cfg.geometry.plate_h_cm,
        },
        "keys": {
            "static": static_keys,
            "frequency": frequency_keys,
            "labels": {
                k: {"title": map_label(k)[0], "unit": _plain(map_label(k)[1])}
                for k in static_keys + frequency_keys
            },
        },
        "conditions": conditions,
    }


def _plain(unit: str) -> str:
    """Strip the LaTeX the matplotlib labels carry; HTML wants plain text."""
    return (
        unit.replace("$", "").replace(r"\cdot", "*").replace(r"\Omega", "ohm")
        .replace(r"\gamma^2", "gamma^2").replace(r"\sigma", "sigma")
        .replace(r"\tau", "tau").replace(r"\chi^2_{red}", "chi2_red")
        .replace("m$\\Omega", "mohm").replace("{", "").replace("}", "")
        .replace("^2", "2").replace("\\", "")
    )


def write_dashboard(
    results: dict[str, ConditionResult], cfg: PipelineConfig, path: str | Path
) -> Path | None:
    """Render the interactive plate view to ``path``."""
    if not results:
        return None
    payload = build_payload(results, cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = _TEMPLATE.replace(
        "__PAYLOAD__", json.dumps(payload, separators=(",", ":"), allow_nan=False)
    )
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Locally-resolved EIS - plate view</title>
<style>
:root{
  --bg:#f7f7f5; --panel:#ffffff; --ink:#1b1c1e; --muted:#6a6f76;
  --line:#e0e0dc; --accent:#3b6ea5; --warn:#b4531f; --plate:#ece7dd;
  --shadow:0 1px 2px rgba(0,0,0,.05),0 4px 14px rgba(0,0,0,.05);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#16171a; --panel:#1e2024; --ink:#e9eaec; --muted:#9aa0a8;
    --line:#2e3238; --accent:#79a9dc; --warn:#e0864a; --plate:#2a2723;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 4px 14px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --bg:#16171a; --panel:#1e2024; --ink:#e9eaec; --muted:#9aa0a8;
  --line:#2e3238; --accent:#79a9dc; --warn:#e0864a; --plate:#2a2723;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 4px 14px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:18px 22px 12px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:17px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:12.5px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
main{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);
  gap:16px;padding:16px 22px 28px;align-items:start}
@media (max-width:1080px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:14px 16px;box-shadow:var(--shadow)}
.controls{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;
  margin-bottom:12px}
label{font-size:12px;color:var(--muted);display:flex;gap:6px;align-items:center}
select,input[type=range]{font:inherit;font-size:13px}
select{background:var(--panel);color:var(--ink);border:1px solid var(--line);
  border-radius:6px;padding:4px 7px}
input[type=range]{accent-color:var(--accent);width:190px}
svg{display:block;width:100%;height:auto;overflow:visible}
.seg{cursor:pointer;transition:opacity .12s}
.seg:hover{stroke:var(--ink);stroke-width:1.6}
.seglabel{font:600 8px ui-monospace,monospace;pointer-events:none;
  fill:#fff;paint-order:stroke;stroke:rgba(0,0,0,.45);stroke-width:2px}
.axis{stroke:var(--line);stroke-width:1}
.tick{fill:var(--muted);font-size:9.5px}
.grid{stroke:var(--line);stroke-dasharray:2 3;stroke-width:.8}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:4px}
th,td{text-align:left;padding:3px 8px 3px 0;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:500}
.pill{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;
  border:1px solid var(--line);color:var(--muted)}
.pill.bad{color:var(--warn);border-color:var(--warn)}
.legendnote{color:var(--muted);font-size:11.5px;margin-top:8px}
.empty{color:var(--muted);font-size:12.5px;padding:20px 0;text-align:center}
</style>
</head>
<body>
<header>
  <h1 id="title">Locally-resolved EIS</h1>
  <div class="sub mono" id="provenance"></div>
</header>
<main>
  <section class="card">
    <div class="controls">
      <label>condition <select id="condition"></select></label>
      <label>map <select id="param"></select></label>
      <label id="freqwrap" hidden>frequency
        <input type="range" id="freq" min="0" max="0" value="0">
        <span class="mono" id="freqval"></span>
      </label>
    </div>
    <svg id="plate" viewBox="0 0 900 470" role="img"
         aria-label="segment map of the plate"></svg>
    <div class="legendnote" id="platenote"></div>
  </section>

  <section class="card">
    <div class="controls">
      <strong id="selname">no segment selected</strong>
      <span class="pill" id="selstatus"></span>
    </div>
    <svg id="nyquist" viewBox="0 0 430 300"></svg>
    <svg id="bode" viewBox="0 0 430 300"></svg>
    <table id="details"></table>
  </section>
</main>

<script>
const DATA = __PAYLOAD__;

/* ---- viridis, sampled every 1/10 and interpolated ------------------- */
const VIRIDIS = [[68,1,84],[72,36,117],[65,68,135],[53,95,141],[42,120,142],
  [33,145,140],[34,168,132],[68,191,112],[122,209,81],[189,223,38],[253,231,37]];
function colour(t){
  if(!isFinite(t)) return "var(--muted)";
  t = Math.min(1, Math.max(0, t));
  const x = t*(VIRIDIS.length-1), i = Math.min(VIRIDIS.length-2, Math.floor(x)),
        u = x-i, a = VIRIDIS[i], b = VIRIDIS[i+1];
  return `rgb(${Math.round(a[0]+u*(b[0]-a[0]))},${Math.round(a[1]+u*(b[1]-a[1]))},`
       + `${Math.round(a[2]+u*(b[2]-a[2]))})`;
}
const el = id => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
function node(parent, name, attrs, text){
  const n = document.createElementNS(SVGNS, name);
  for(const k in attrs) if(attrs[k]!==null && attrs[k]!==undefined)
    n.setAttribute(k, attrs[k]);
  if(text!==undefined) n.textContent = text;
  parent.appendChild(n); return n;
}
const fmt = (v, d=3) => (v===null||v===undefined||!isFinite(v)) ? "-"
  : (Math.abs(v)>=1e4||(Math.abs(v)<1e-3&&v!==0) ? v.toExponential(2)
     : v.toFixed(d).replace(/\.?0+$/,""));

/* ---- state ---------------------------------------------------------- */
let condition = Object.keys(DATA.conditions)[0];
let param = (DATA.keys.static[0] || DATA.keys.frequency[0]);
let freqIndex = 0;
let selected = null;

/* ---- frequency grid shared by the condition ------------------------- */
function grid(){
  const segs = DATA.conditions[condition].segments;
  for(const k in segs) if(segs[k].f && segs[k].f.length) return segs[k].f;
  return [];
}

/* ---- the value a segment contributes to the current map ------------- */
function valueOf(key, seg, fi){
  const c = DATA.conditions[condition];
  if(DATA.keys.frequency.includes(key)){
    const s = c.segments[seg];
    if(!s || !s.f || fi>=s.f.length) return NaN;
    const re = s.zr[fi], im = s.zi[fi];
    if(re===null||im===null) return NaN;
    switch(key){
      case "z_mod@f":     return Math.hypot(re, im);
      case "z_real@f":    return re;
      case "neg_z_imag@f":return -im;
      case "phase@f":     return Math.atan2(im, re)*180/Math.PI;
      case "coherence@f": return s.coh[fi];
    }
    return NaN;
  }
  const m = c.maps[key];
  const v = m ? m[seg] : undefined;
  return (v===undefined||v===null) ? NaN : v;
}

/* ---- plate ---------------------------------------------------------- */
function drawPlate(){
  const svg = el("plate"); svg.textContent = "";
  const c = DATA.conditions[condition], segs = c.segments;
  const W = 900, H = 470, padL = 12, padR = 118, padT = 10, padB = 26;
  const pw = DATA.plate.w || 1, ph = DATA.plate.h || 1;
  const scale = Math.min((W-padL-padR)/pw, (H-padT-padB)/ph);
  const ox = padL + ((W-padL-padR) - pw*scale)/2;
  const oy = padT + ((H-padT-padB) - ph*scale)/2;
  const X = x => ox + x*scale, Y = y => oy + (ph - y)*scale;

  node(svg,"rect",{x:X(0),y:Y(ph),width:pw*scale,height:ph*scale,
    fill:"var(--plate)",stroke:"var(--line)","stroke-width":1.5,rx:4});

  /* Colour scale from the segments a reader should trust, so one broken
     channel cannot flatten the whole map - but every segment is still drawn. */
  const trusted = [], all = [];
  for(const k in segs){
    const v = valueOf(param, k, freqIndex);
    if(!isFinite(v)) continue;
    all.push(v);
    if(segs[k].active && segs[k].quality >= DATA.meta.good_quality) trusted.push(v);
  }
  const basis = trusted.length >= 3 ? trusted : all;
  let lo = Math.min(...basis), hi = Math.max(...basis);
  if(!isFinite(lo)||!isFinite(hi)||lo===hi){ lo = (lo||0)-1; hi = (hi||0)+1; }

  const keys = Object.keys(segs).sort((a,b)=>(+a)-(+b));
  for(const k of keys){
    const s = segs[k];
    if(!s.coords) continue;
    const [x,y,hw,hh] = s.coords, v = valueOf(param, k, freqIndex);
    const good = s.active && s.quality >= DATA.meta.good_quality;
    const g = node(svg,"g",{class:"seg","data-seg":k});
    node(g,"rect",{
      x:X(x-hw), y:Y(y+hh), width:2*hw*scale, height:2*hh*scale, rx:2.5,
      fill: isFinite(v) ? colour((v-lo)/(hi-lo)) : "none",
      "fill-opacity": !s.active ? 0.12 : (good ? 1 : 0.45),
      stroke: selected===k ? "var(--ink)" : (good ? "#fff" : "var(--warn)"),
      "stroke-width": selected===k ? 2.4 : 1.1,
      "stroke-dasharray": good ? null : "3 2"
    });
    // Label size follows the box, so an 80-segment plate stays readable and a
    // 20-segment one does not look like large-print.
    const boxH = 2*hh*scale, boxW = 2*hw*scale;
    const size = Math.max(5, Math.min(11, boxH/5, boxW/4.5));
    node(g,"text",{x:X(x), y:Y(y)-size*0.15, "text-anchor":"middle",
      class:"seglabel", "font-size":size}, k);
    node(g,"text",{x:X(x), y:Y(y)+size*1.05, "text-anchor":"middle",
      class:"seglabel", "font-size":size*0.92},
      s.active ? fmt(v,2) : s.status);
    node(g,"title",{},
      `segment ${k} - card ${s.card}\n${DATA.keys.labels[param].title} = `
      + `${fmt(v,4)} ${DATA.keys.labels[param].unit}\nstatus ${s.status}, `
      + `quality ${fmt(s.quality,2)}` + (s.flags.length ? `\n${s.flags.join(", ")}`:""));
    g.addEventListener("click", () => { selected = k; render(); });
  }

  /* colour bar */
  const bx = W - padR + 26, by = oy, bh = ph*scale;
  const gradId = "grad";
  const defs = node(svg,"defs",{});
  const lg = node(defs,"linearGradient",{id:gradId,x1:0,y1:1,x2:0,y2:0});
  for(let i=0;i<=10;i++)
    node(lg,"stop",{offset:(i*10)+"%","stop-color":colour(i/10)});
  node(svg,"rect",{x:bx,y:by,width:16,height:bh,fill:`url(#${gradId})`,
    stroke:"var(--line)"});
  for(let i=0;i<=4;i++){
    const v = lo + (hi-lo)*i/4;
    node(svg,"text",{x:bx+21,y:by+bh-(bh*i/4)+3.5,class:"tick"}, fmt(v,3));
  }
  node(svg,"text",{x:bx-4,y:by-4,class:"tick","text-anchor":"end"},
    DATA.keys.labels[param].unit || DATA.keys.labels[param].title);

  const counts = Object.entries(c.status_counts)
    .map(([k,v])=>`${v} ${k}`).join(", ");
  el("platenote").textContent =
    `${c.n_active}/${c.n_total} segments active (${counts}). `
    + `Dashed and desaturated: quality below ${DATA.meta.good_quality}. `
    + `Outline only: no usable frequency point. Click a segment for its spectrum.`;
}

/* ---- spectra -------------------------------------------------------- */
function axes(svg, box, xr, yr, xlabel, ylabel, logx, title){
  const {l,r,t,b,W,H} = box;
  const X = v => l + (( logx ? (Math.log10(v)-Math.log10(xr[0]))
      /(Math.log10(xr[1])-Math.log10(xr[0])) : (v-xr[0])/(xr[1]-xr[0]) ))*(W-l-r);
  const Y = v => (H-b) - ((v-yr[0])/(yr[1]-yr[0]))*(H-t-b);
  for(let i=0;i<=4;i++){
    const yv = yr[0]+(yr[1]-yr[0])*i/4;
    node(svg,"line",{x1:l,x2:W-r,y1:Y(yv),y2:Y(yv),class:"grid"});
    node(svg,"text",{x:l-5,y:Y(yv)+3.5,class:"tick","text-anchor":"end"},fmt(yv,3));
  }
  for(let i=0;i<=4;i++){
    const xv = logx
      ? Math.pow(10, Math.log10(xr[0])+(Math.log10(xr[1])-Math.log10(xr[0]))*i/4)
      : xr[0]+(xr[1]-xr[0])*i/4;
    node(svg,"line",{x1:X(xv),x2:X(xv),y1:t,y2:H-b,class:"grid"});
    node(svg,"text",{x:X(xv),y:H-b+13,class:"tick","text-anchor":"middle"},fmt(xv,3));
  }
  node(svg,"line",{x1:l,x2:W-r,y1:H-b,y2:H-b,class:"axis"});
  node(svg,"line",{x1:l,x2:l,y1:t,y2:H-b,class:"axis"});
  node(svg,"text",{x:(l+W-r)/2,y:H-2,class:"tick","text-anchor":"middle"},xlabel);
  // Rotated along the left edge: a horizontal y-label long enough to name a
  // unit runs straight through the chart title.
  node(svg,"text",{x:9,y:(t+H-b)/2,class:"tick","text-anchor":"middle",
    transform:`rotate(-90 9 ${(t+H-b)/2})`}, ylabel);
  if(title) node(svg,"text",{x:(l+W-r)/2,y:t-3,class:"tick",
    "text-anchor":"middle"}, title);
  return {X,Y};
}

function drawSpectra(){
  const ny = el("nyquist"), bo = el("bode");
  ny.textContent = ""; bo.textContent = "";
  const s = selected ? DATA.conditions[condition].segments[selected] : null;
  if(!s || !s.f || !s.f.length){
    node(ny,"text",{x:215,y:150,class:"tick","text-anchor":"middle"},
      selected ? "this segment produced no spectrum" : "click a segment on the plate");
    return;
  }
  const idx = s.f.map((_,i)=>i).filter(i => s.zr[i]!==null);
  const re = idx.map(i=>s.zr[i]), im = idx.map(i=>-s.zi[i]);
  const err = idx.map(i=>Math.hypot(s.zr[i],s.zi[i])*(s.sig[i]||0));

  /* Nyquist */
  {
    const box={l:52,r:12,t:22,b:26,W:430,H:300};
    const pad = v => { const a=Math.min(...v),b=Math.max(...v),m=(b-a||1)*0.12;
      return [a-m,b+m]; };
    const A = axes(ny,box,pad(re),pad(im),"Z' [mohm*cm2]","-Z'' [mohm*cm2]",false,
      "Nyquist" + (s.fr ? ` (dashed = ${s.model} fit)` : ""));
    if(s.fr){
      const p = idx.map(i=>`${A.X(s.fr[i])},${A.Y(-s.fi[i])}`).join(" ");
      node(ny,"polyline",{points:p,fill:"none",stroke:"var(--muted)",
        "stroke-width":1.4,"stroke-dasharray":"4 3"});
    }
    node(ny,"polyline",{points:idx.map((i,j)=>`${A.X(re[j])},${A.Y(im[j])}`).join(" "),
      fill:"none",stroke:"var(--accent)","stroke-width":1.3});
    idx.forEach((i,j)=>{
      node(ny,"line",{x1:A.X(re[j]),x2:A.X(re[j]),y1:A.Y(im[j]-err[j]),
        y2:A.Y(im[j]+err[j]),stroke:"var(--accent)","stroke-width":.8,opacity:.6});
      node(ny,"circle",{cx:A.X(re[j]),cy:A.Y(im[j]),r:s.used[i]?2.8:2,
        fill:s.used[i]?"var(--accent)":"none",stroke:"var(--accent)",
        "stroke-width":1});
      node(ny,"title",{},`${fmt(s.f[i],4)} Hz\nZ = ${fmt(re[j],4)} `
        + `${im[j]>=0?"-":"+"} ${fmt(Math.abs(im[j]),4)}j mohm*cm2\n`
        + `gamma2 ${fmt(s.coh[i],3)}, sigma ${fmt(100*(s.sig[i]||0),2)}%`
        + (s.used[i]?"":"\nbelow the coherence gate - shown, not used"));
    });
  }

  /* Bode: |Z| and phase */
  {
    const box={l:52,r:40,t:22,b:26,W:430,H:300};
    const mod = idx.map(i=>Math.hypot(s.zr[i],s.zi[i]));
    const pha = idx.map(i=>Math.atan2(s.zi[i],s.zr[i])*180/Math.PI);
    const fr = [Math.min(...s.f.filter(v=>v>0)), Math.max(...s.f)];
    const A = axes(bo,box,fr,[Math.min(...mod)*0.9,Math.max(...mod)*1.1],
      "f [Hz]","|Z| [mohm*cm2]",true,
      "Bode - magnitude (solid) and phase (dashed, right axis)");
    node(bo,"polyline",{points:idx.map((i,j)=>`${A.X(s.f[i])},${A.Y(mod[j])}`).join(" "),
      fill:"none",stroke:"var(--accent)","stroke-width":1.4});
    const pr=[Math.min(...pha),Math.max(...pha)], span=(pr[1]-pr[0])||1;
    const PY = v => (box.H-box.b) - ((v-pr[0])/span)*(box.H-box.t-box.b);
    node(bo,"polyline",{points:idx.map((i,j)=>`${A.X(s.f[i])},${PY(pha[j])}`).join(" "),
      fill:"none",stroke:"var(--warn)","stroke-width":1.2,"stroke-dasharray":"3 2"});
    for(let i=0;i<=4;i++){
      const v = pr[0]+span*i/4;
      node(bo,"text",{x:box.W-box.r+4,y:PY(v)+3.5,class:"tick"},fmt(v,1));
    }
    node(bo,"text",{x:box.W-9,y:(box.t+box.H-box.b)/2,class:"tick",
      "text-anchor":"middle",
      transform:`rotate(90 ${box.W-9} ${(box.t+box.H-box.b)/2})`},"phase [deg]");
    idx.forEach((i,j)=>node(bo,"circle",{cx:A.X(s.f[i]),cy:A.Y(mod[j]),
      r:s.used[i]?2.6:1.8,fill:s.used[i]?"var(--accent)":"none",
      stroke:"var(--accent)","stroke-width":1}));
  }
}

function drawDetails(){
  const t = el("details"); t.textContent = "";
  const c = DATA.conditions[condition];
  const s = selected ? c.segments[selected] : null;
  const rows = [];
  if(s){
    const cm = c.common_mode[String(s.card)];
    rows.push(["card", s.card],
      ["status", s.status],
      ["quality", fmt(s.quality,2)],
      ["Rs (HF fit)", `${fmt(s.rs,3)} mohm*cm2`],
      ["KK max residual", s.kk===null ? "-" : `${fmt(s.kk,2)} %`],
      ["points used", `${s.used.reduce((a,b)=>a+b,0)} of ${s.f.length}`],
      ["physical units", s.physical ? "yes" : "no - shunt volts per amp"],
      ["flags", s.flags.length ? s.flags.join(", ") : "-"]);
    if(cm) rows.push(["card common-mode delay",
      `${fmt(cm.delay_ns,1)} ns ${cm.applied?"(applied)":"(not applied)"}`]);
    if(s.note) rows.push(["note", s.note]);
  } else {
    rows.push(["condition", condition],
      ["segments active", `${c.n_active} of ${c.n_total}`],
      ["record length", `${c.duration_s} s`],
      ["reference card", c.reference_card],
      ["synchronisation", c.sync_passed ? "pass" : "FAIL"]);
    if(c.tone_check) rows.push(["excitation", c.tone_check]);
  }
  for(const [k,v] of rows){
    const tr = document.createElement("tr");
    const th = document.createElement("th"); th.textContent = k;
    const td = document.createElement("td"); td.textContent = v;
    td.className = "mono"; tr.appendChild(th); tr.appendChild(td);
    t.appendChild(tr);
  }
  el("selname").textContent = s ? `segment ${selected}` : "plate overview";
  const badge = el("selstatus");
  badge.textContent = s ? s.status : "";
  badge.className = "pill" + (s && s.status !== "ok" ? " bad" : "");
}

function render(){
  const isFreq = DATA.keys.frequency.includes(param);
  const g = grid();
  el("freqwrap").hidden = !isFreq || !g.length;
  if(isFreq && g.length){
    const slider = el("freq");
    slider.max = g.length - 1;
    freqIndex = Math.min(freqIndex, g.length - 1);
    slider.value = freqIndex;
    el("freqval").textContent = `${fmt(g[freqIndex],4)} Hz`;
  }
  drawPlate(); drawSpectra(); drawDetails();
}

/* ---- wiring --------------------------------------------------------- */
(function init(){
  el("title").textContent =
    `Locally-resolved EIS - ${DATA.meta.measurement_id || "measurement"}`;
  el("provenance").textContent =
    `pipeline ${DATA.meta.pipeline_version} | param_hash ${DATA.meta.param_hash}`
    + ` | git ${DATA.meta.git_sha} | ${DATA.meta.created_utc}`;

  const cs = el("condition");
  for(const name of Object.keys(DATA.conditions)){
    const o = document.createElement("option");
    o.value = name; o.textContent = name; cs.appendChild(o);
  }
  cs.value = condition;
  cs.addEventListener("change", e => { condition = e.target.value; render(); });

  const ps = el("param");
  for(const key of DATA.keys.static.concat(DATA.keys.frequency)){
    const o = document.createElement("option");
    o.value = key; o.textContent = DATA.keys.labels[key].title; ps.appendChild(o);
  }
  ps.value = param;
  ps.addEventListener("change", e => { param = e.target.value; render(); });

  el("freq").addEventListener("input", e => {
    freqIndex = +e.target.value; render();
  });
  render();
})();
</script>
</body>
</html>
"""
