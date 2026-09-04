"""Write the interactive plate map as ONE self-contained HTML file.

`plate_model.draw_plate` gives a PNG and `plate_plotly` needs the Dash server.
This gives the third thing: a single `.html` your evaluation drops next to its
other outputs, that opens in any browser with every control the web mock-up has
-- field switching, cathode/anode side, flow animation, channels, numbering,
segment numbers, hover and click-to-pin -- and no server, no network, no
dependencies.

    from plate_model import PLATE, synthetic_fields
    from plate_viewer import write_html, Field

    fields = [
        Field("j",   "Current density",   "A/cm²",   "inferno", 3, values_j),
        Field("hfr", "HFR (Rs)",          "mΩ·cm²",  "viridis", 1, values_hfr),
        Field("rp",  "Rp",                "mΩ·cm²",  "magma",   0, values_rp),
    ]
    write_html(fields, "out/plate.html", title="2611976 · 150 A")

`values_*` are plain dicts keyed by segment number in whichever scheme you pass
as `scheme=`.  Anything missing is drawn grey rather than left as a hole -- a
hole in a heat map reads to the eye as a cold spot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

# Imported both ways: as part of the app package, and as a loose
# script (`python plate_viewer.py`) by someone who only wants the
# standalone HTML. A bare relative import breaks the second, a bare
# absolute one breaks the first.
try:
    from .plate_model import (N_COLS, N_ROWS, PLATE, PORTS, TEMP_SENSOR_X_MM, Plate,
                             robust_limits)
except ImportError:                       # pragma: no cover
    from plate_model import (N_COLS, N_ROWS, PLATE, PORTS, TEMP_SENSOR_X_MM, Plate,
                             robust_limits)

#: The colour ramps the viewer knows, as control points.  Keep them
#: perceptually ordered -- a rainbow invents structure that is not in the data.
RAMPS: Dict[str, Sequence[str]] = {
    "inferno": ["#040414", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"],
    "thermal": ["#313695", "#74add1", "#e0f3f8", "#ffffbf", "#fee090", "#f46d43", "#a50026"],
    "viridis": ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
    "humid":   ["#ffffd9", "#c7e9b4", "#41b6c4", "#225ea8", "#081d58"],
    "magma":   ["#040414", "#51127c", "#b73779", "#fc8961", "#fcfdbf"],
    "cividis": ["#00224e", "#35456c", "#666970", "#948e64", "#d9c65b", "#fee838"],
    "water":   ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
}


@dataclass
class Field:
    """One mappable quantity: what it is called, how it is drawn, its values."""
    key: str
    label: str
    unit: str
    ramp: str
    decimals: int
    values: Mapping[int, float]
    note: str = ""


def _payload(fields: Sequence[Field], scheme: str, plate: Plate) -> dict:
    segments = []
    for k in sorted(plate.segments):
        seg = plate.segments[k]
        rings = seg.rings_mm()
        d = " ".join("M" + "L".join(f"{x:.3f} {y:.3f}" for x, y in ring[:-1]) + "Z"
                     for ring in rings)
        lx, ly = seg.label_point_mm
        cx, cy = seg.centroid_mm
        segments.append(dict(wired=seg.wired, topleft=seg.topleft, d=d,
                             lx=round(lx, 2), ly=round(ly, 2),
                             cx=round(cx, 2), cy=round(cy, 2),
                             np=seg.n_pads, area=round(seg.area_cm2, 4)))

    out_fields = []
    for f in fields:
        by_wired, raw = {}, []
        for k, seg in plate.segments.items():
            key = k if scheme == "wired" else seg.topleft
            v = f.values.get(key)
            v = float(v) if v is not None and np.isfinite(v) else None
            by_wired[str(k)] = v
            if v is not None:
                raw.append(v)
        lo, hi = robust_limits(raw)
        out_fields.append(dict(key=f.key, label=f.label, unit=f.unit,
                               ramp=f.ramp if f.ramp in RAMPS else "viridis",
                               dec=f.decimals, note=f.note,
                               lo=lo, hi=hi if hi > lo else lo + 1e-12,
                               values=by_wired))

    return dict(
        w=plate.w_mm, h=plate.h_mm, area=plate.area_cm2,
        cols=N_COLS, rows=N_ROWS, scheme=scheme,
        sensors=[dict(name=n.replace("temp", "T"), x=x)
                 for n, x in TEMP_SENSOR_X_MM.items()],
        ports=[dict(key=p.key, label=p.label, side=p.side, corner=p.corner,
                    flow=p.flow, rect=list(p.rect), arrow=list(p.arrow),
                    color=p.color) for p in PORTS],
        segments=segments, fields=out_fields, ramps=RAMPS)


def write_html(fields: Sequence[Field], path: str | Path,
               title: str = "Segmented plate", subtitle: str = "",
               scheme: str = "wired", plate: Plate = PLATE) -> Path:
    """Write the viewer.  Returns the path written."""
    if not fields:
        raise ValueError("write_html needs at least one Field")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(_payload(fields, scheme, plate), separators=(",", ":"))
    path.write_text(_TEMPLATE.replace("__TITLE__", title)
                             .replace("__SUBTITLE__", subtitle or plate.describe())
                             .replace("__DATA__", data), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600&family=Barlow:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#f2f2f3; --text:#1d1f20; --accent:#5980a6; --accent-700:#416180;
  --n600:#7a7a7d; --n700:#5d5d60; --n800:#424244; --divider:rgba(29,31,32,.16);
  --head:"Barlow Condensed",system-ui,sans-serif; --body:"Barlow",system-ui,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--body);
     padding:26px 30px 44px;max-width:1720px;margin:0 auto}
h1{font-family:var(--head);font-weight:600;font-size:44px;line-height:1;margin:6px 0 0}
.kicker{font-family:var(--head);font-weight:600;font-size:13px;letter-spacing:.14em;
        text-transform:uppercase;color:var(--accent-700)}
.sub{font-size:14px;color:var(--n700);margin-top:7px;max-width:80ch;text-wrap:pretty}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;
       flex-wrap:wrap;border-bottom:1px solid var(--divider);padding-bottom:14px;margin-bottom:18px}
.meta{font-family:var(--mono);font-size:12px;color:var(--n700);text-align:right;line-height:1.6}
.bar{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-end;margin-bottom:18px}
.grp>div:first-child{font-family:var(--head);font-size:12px;letter-spacing:.12em;
                     text-transform:uppercase;color:var(--n700);margin-bottom:7px}
.grp>div:last-child{display:flex;gap:6px;flex-wrap:wrap}
button{font-family:var(--body);font-size:12px;padding:7px 12px;border-radius:0;
       border:1px solid var(--accent);background:transparent;color:var(--accent-700);
       cursor:pointer;transition:background .12s}
button:hover{background:color-mix(in srgb,var(--accent) 12%,transparent)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
button.on:hover{background:var(--accent-700)}
main{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:18px;align-items:start}
.card{position:relative;border:1px solid var(--divider);padding:14px 16px;background:transparent}
.card>i{position:absolute;width:9px;height:9px;color:var(--accent);opacity:.75}
.card>i::before{content:"+";position:absolute;inset:-7px 0 0 -3px;font:12px/1 var(--mono)}
.card>i.tl{left:-1px;top:-1px}.card>i.tr{right:-1px;top:-1px}
.card>i.bl{left:-1px;bottom:-1px}.card>i.br{right:-1px;bottom:-1px}
.rail{display:flex;flex-direction:column;gap:14px}
.rowk{display:flex;justify-content:space-between;gap:10px;font-size:12.5px;
      border-bottom:1px solid var(--divider);padding-bottom:3px}
.rowk span:first-child{color:var(--n700)}
.rowk span:last-child{font-family:var(--mono);font-weight:600}
.cap{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-top:8px;
     font-size:11.5px;color:var(--n700);font-family:var(--mono)}
@keyframes fd{to{stroke-dashoffset:-60}}
@keyframes fdr{to{stroke-dashoffset:60}}
.fl{animation:fd 2.6s linear infinite}
.fl.rev{animation-name:fdr}
.seg{cursor:pointer}
.seg:hover{filter:brightness(1.14)}
@media print{button{display:none}main{grid-template-columns:1fr}}
</style></head>
<body>
<header>
  <div>
    <div class="kicker">Local EIS · segmented plate</div>
    <h1>__TITLE__</h1>
    <div class="sub">__SUBTITLE__</div>
  </div>
  <div class="meta" id="meta"></div>
</header>

<div class="bar">
  <div class="grp" style="flex:1 1 520px"><div>Field</div><div id="c-field"></div></div>
  <div class="grp"><div>Side shown</div><div id="c-side"></div></div>
  <div class="grp"><div>Overlay</div><div id="c-over"></div></div>
  <div class="grp"><div>Numbering</div><div id="c-num"></div></div>
</div>

<main>
  <div class="card"><i class="tl"></i><i class="tr"></i><i class="bl"></i><i class="br"></i>
    <svg id="plate" viewBox="-42 -54 336 240" style="width:100%;height:auto;display:block"></svg>
    <div class="cap"><span id="cap"></span><span id="src"></span></div>
  </div>
  <div class="rail">
    <div class="card"><i class="tl"></i><i class="tr"></i><i class="bl"></i><i class="br"></i>
      <div style="font-family:var(--head);font-weight:600;font-size:17px;line-height:1.1" id="fl"></div>
      <div style="font-size:12px;color:var(--n700);margin-top:2px;font-family:var(--mono)" id="fu"></div>
      <div style="display:flex;gap:10px;margin-top:12px">
        <svg id="cbar" viewBox="0 0 12 120" style="width:26px;height:150px;flex:none"></svg>
        <div id="cticks" style="display:flex;flex-direction:column;justify-content:space-between;
             height:150px;font-family:var(--mono);font-size:11.5px;color:var(--n800)"></div>
      </div>
      <div style="font-size:12px;color:var(--n700);margin-top:10px;line-height:1.45" id="fnote"></div>
    </div>
    <div class="card"><i class="tl"></i><i class="tr"></i><i class="bl"></i><i class="br"></i>
      <div style="font-family:var(--head);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--n700)">Segment</div>
      <div style="display:flex;align-items:baseline;gap:8px;margin-top:2px">
        <div style="font-family:var(--head);font-weight:600;font-size:34px;line-height:1" id="it">—</div>
        <div style="font-size:12px;color:var(--n700);font-family:var(--mono)" id="is">hover the plate</div>
      </div>
      <div style="display:flex;flex-direction:column;gap:4px;margin-top:12px" id="irows"></div>
      <div style="font-size:11.5px;color:var(--n600);margin-top:10px">Hover to preview, click to pin.</div>
    </div>
    <div class="card"><i class="tl"></i><i class="tr"></i><i class="bl"></i><i class="br"></i>
      <div style="font-family:var(--head);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--n700);margin-bottom:8px">Distribution · 72 segments</div>
      <svg id="dist" viewBox="0 0 220 62" style="width:100%;height:auto;display:block"></svg>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--n700);font-family:var(--mono);margin-top:4px">
        <span>sorted low → high</span><span id="spread"></span></div>
    </div>
    <div class="card"><i class="tl"></i><i class="tr"></i><i class="bl"></i><i class="br"></i>
      <div style="font-family:var(--head);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--n700);margin-bottom:8px">Ports</div>
      <div id="portlist" style="display:flex;flex-direction:column;gap:6px;font-size:12.5px"></div>
      <div style="font-size:11.5px;color:var(--n600);margin-top:10px;line-height:1.45">
        Counter-flow: the gases traverse the plate in opposite directions, so each inlet faces the other's outlet.</div>
    </div>
  </div>
</main>

<script>
const D = __DATA__;
const S = {field:D.fields[0].key, side:"cathode", labels:true, flow:true,
           channels:true, wired:D.scheme==="wired", hover:null, pinned:null};
const NS="http://www.w3.org/2000/svg";
const el=(t,a,p)=>{const n=document.createElementNS(NS,t);
  for(const k in a||{}) n.setAttribute(k,a[k]); if(p)p.appendChild(n); return n;};

function hex2rgb(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function ramp(name,t){const cs=D.ramps[name]||D.ramps.viridis;
  t=Math.max(0,Math.min(1,t)); const p=t*(cs.length-1),i=Math.min(cs.length-2,Math.floor(p)),f=p-i;
  const a=hex2rgb(cs[i]),b=hex2rgb(cs[i+1]);
  return "rgb("+a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(",")+")";}
function ink(rgb){const m=rgb.match(/\d+/g).map(Number).map(c=>{const s=c/255;
  return s<=0.04045?s/12.92:Math.pow((s+0.055)/1.055,2.4);});
  return (0.2126*m[0]+0.7152*m[1]+0.0722*m[2])>0.42?"rgba(20,20,20,.88)":"rgba(255,255,255,.95)";}
const F=()=>D.fields.find(f=>f.key===S.field)||D.fields[0];
const fmt=(v,f)=>v==null?"—":v.toFixed(f.dec);
const num=s=>S.wired?s.wired:s.topleft;

function chip(host,label,on,fn){const b=document.createElement("button");
  b.textContent=label; if(on)b.className="on"; b.onclick=()=>{fn();draw();}; host.appendChild(b);}

function controls(){
  const f=document.getElementById("c-field"); f.innerHTML="";
  D.fields.forEach(x=>chip(f,x.label,S.field===x.key,()=>S.field=x.key));
  const s=document.getElementById("c-side"); s.innerHTML="";
  chip(s,"Cathode · air",S.side==="cathode",()=>S.side="cathode");
  chip(s,"Anode · H₂",S.side==="anode",()=>S.side="anode");
  const o=document.getElementById("c-over"); o.innerHTML="";
  chip(o,"Numbers",S.labels,()=>S.labels=!S.labels);
  chip(o,"Flow",S.flow,()=>S.flow=!S.flow);
  chip(o,"Channels",S.channels,()=>S.channels=!S.channels);
  const n=document.getElementById("c-num"); n.innerHTML="";
  chip(n,"As wired",S.wired,()=>S.wired=true);
  chip(n,"Top-left down",!S.wired,()=>S.wired=false);
}

function sideArrow(x0,x1,y,w,hw,hl){const d=x1>x0?1:-1,n=x1-d*hl;
  return `M${x0} ${y-w}L${n} ${y-w}L${n} ${y-hw}L${x1} ${y}L${n} ${y+hw}L${n} ${y+w}L${x0} ${y+w}Z`;}

function draw(){
  controls();
  const f=F(), W=D.w, H=D.h, svg=document.getElementById("plate");
  svg.innerHTML="";
  const defs=el("defs",{},svg);
  const g1=el("linearGradient",{id:"metal",x1:"0",y1:"0",x2:"0.7",y2:"1"},defs);
  [["0","#e4e6e9"],["0.34","#f4f5f7"],["0.62","#d3d6da"],["1","#bfc3c8"]]
    .forEach(([o,c])=>el("stop",{offset:o,"stop-color":c},g1));
  const g2=el("linearGradient",{id:"act",x1:"0",y1:"0",x2:"0.5",y2:"1"},defs);
  [["0","#9aa0a7"],["0.5","#b6bcc3"],["1","#8c9299"]]
    .forEach(([o,c])=>el("stop",{offset:o,"stop-color":c},g2));
  const pat=el("pattern",{id:"mill",width:"6",height:"1.55",patternUnits:"userSpaceOnUse"},defs);
  el("rect",{width:"6",height:"1.55",fill:"#a9afb6"},pat);
  el("rect",{width:"6",height:"0.78",fill:"#7f858c"},pat);
  const gk=el("pattern",{id:"gk",width:"3",height:"3",patternUnits:"userSpaceOnUse",
                         patternTransform:"rotate(45)"},defs);
  el("line",{x1:"0",y1:"0",x2:"0",y2:"3",stroke:"#5d5d60","stroke-width":"0.45",opacity:".35"},gk);
  const cp=el("clipPath",{id:"ac"},defs); el("rect",{x:"0",y:"0",width:W,height:H},cp);
  const sh=el("filter",{id:"sh",x:"-20%",y:"-20%",width:"140%",height:"140%"},defs);
  el("feDropShadow",{dx:"0",dy:"2.4",stdDeviation:"2.6","flood-color":"#2b2b2d",
                     "flood-opacity":"0.28"},sh);

  el("rect",{x:-24,y:-28,width:300,height:177,rx:7,fill:"url(#metal)",stroke:"#8b9097",
             "stroke-width":.6,filter:"url(#sh)"},svg);
  el("rect",{x:-18.5,y:-22.5,width:289,height:166,rx:4,fill:"none",stroke:"#8b9097",
             "stroke-width":.35,opacity:.65},svg);
  el("rect",{x:-6,y:-10,width:264,height:141,rx:3,fill:"url(#gk)",stroke:"#6f747b","stroke-width":.5},svg);
  el("rect",{x:0,y:0,width:W,height:H,fill:"url(#act)"},svg);
  if(S.channels){const c=el("g",{"clip-path":"url(#ac)"},svg);
    el("rect",{x:0,y:0,width:W,height:H,fill:"url(#mill)"},c);}

  // field
  const vals=Object.values(f.values).filter(v=>v!=null).sort((a,b)=>a-b);
  const fg=el("g",{opacity:.88},svg);
  D.segments.forEach(s=>{
    const v=f.values[s.wired];
    const c=v==null?"#d4d4d7":ramp(f.ramp,(v-f.lo)/(f.hi-f.lo));
    const on=(S.pinned??S.hover)===s.wired;
    const p=el("path",{class:"seg",d:s.d,fill:c,stroke:on?"#1d1f20":"rgba(29,31,32,.55)",
                       "stroke-width":on?1.4:.35},fg);
    const t=el("title",{},p);
    t.textContent=`Segment ${s.wired} (as wired) · ${s.topleft} (top-left down)\n`+
      `${f.label}: ${fmt(v,f)} ${f.unit}\n${s.area.toFixed(3)} cm², ${s.np} pads`;
    p.onmouseenter=()=>{S.hover=s.wired;inspector();};
    p.onclick=()=>{S.pinned=S.pinned===s.wired?null:s.wired;draw();};
  });
  svg.onmouseleave=()=>{S.hover=null;inspector();};

  // flow
  if(S.flow){const fgp=el("g",{"clip-path":"url(#ac)","pointer-events":"none"},svg);
    const an=S.side==="anode";
    for(let i=0;i<13;i++){const y=5.5+i*9.2;
      const l=el("line",{class:"fl"+(an?" rev":""),x1:-3,x2:W+3,y1:y,y2:y,
        stroke:an?"rgba(255,236,230,.9)":"rgba(232,244,255,.9)","stroke-width":1.6,
        "stroke-linecap":"round","stroke-dasharray":"8 14"},fgp);
      l.style.animationDelay=(-i*0.19).toFixed(2)+"s";}}

  // numbers
  if(S.labels){const lg=el("g",{"pointer-events":"none"},svg);
    D.segments.forEach(s=>{const v=f.values[s.wired];
      const c=v==null?"#d4d4d7":ramp(f.ramp,(v-f.lo)/(f.hi-f.lo));
      const fo=el("foreignObject",{x:s.lx-8,y:s.ly-3.5,width:16,height:7},lg);
      const dv=document.createElement("div");
      dv.style.cssText="font:600 4.3px var(--mono);text-align:center;line-height:7px;color:"+
        (v==null?"rgba(20,20,20,.88)":ink(c));
      dv.textContent=num(s); fo.appendChild(dv);});}

  el("rect",{x:0,y:0,width:W,height:H,fill:"none",stroke:"#5d5d60","stroke-width":.7,
             "pointer-events":"none"},svg);

  // sensors
  const sg=el("g",{"pointer-events":"none"},svg);
  D.sensors.forEach(t=>{
    el("line",{x1:t.x,y1:-9,x2:t.x,y2:-1.5,stroke:"#1d1f20","stroke-width":.6},sg);
    el("circle",{cx:t.x,cy:-10.6,r:1.7,fill:"#f2f2f3",stroke:"#1d1f20","stroke-width":.6},sg);
    const fo=el("foreignObject",{x:t.x-8,y:-18,width:16,height:5},sg);
    const dv=document.createElement("div");
    dv.style.cssText="font:600 3.3px var(--mono);text-align:center;line-height:5px";
    dv.textContent=t.name; fo.appendChild(dv);});

  // ports + arrows
  D.ports.forEach(p=>{
    const live=p.side===S.side?1:.3, [x,y,w,h]=p.rect, [a0,a1,ay]=p.arrow;
    const g=el("g",{opacity:live},svg);
    el("rect",{x:x,y:y,width:w,height:h,rx:4,fill:"#3b3f44"},g);
    el("rect",{x:x+2,y:y+2,width:w-4,height:h-4,rx:3,fill:"#25282c"},g);
    el("rect",{x:x,y:y,width:w,height:h,rx:4,fill:"none",stroke:p.color,"stroke-width":1.1},g);
    el("path",{d:sideArrow(a0,a1,ay,2.4,6.2,9),fill:p.color},g);
    const left=p.corner[1]==="l";
    const fo=el("foreignObject",{x:left?50:82,y:p.corner[0]==="t"?-51:149,width:120,height:18},g);
    const d1=document.createElement("div");
    d1.style.cssText=`font:600 7.4px var(--head);letter-spacing:.06em;line-height:8.6px;
      text-align:${left?"left":"right"};color:${p.color}`;
    d1.textContent=p.label;
    const d2=document.createElement("div");
    d2.style.cssText=`font:3.4px var(--mono);line-height:4.4px;text-align:${left?"left":"right"};color:#5d5d60`;
    d2.textContent=(p.side==="anode"?"anode ":"cathode ")+(p.flow==="in"?"inlet":"outlet");
    fo.appendChild(d1); fo.appendChild(d2);});

  // bolts
  const bg=el("g",{"pointer-events":"none"},svg);
  const bolt=(x,y)=>{el("circle",{cx:x,cy:y,r:2.6,fill:"#b7b7ba",stroke:"#7a7a7d","stroke-width":.4},bg);
                     el("circle",{cx:x,cy:y,r:1.5,fill:"#4a4d52"},bg);};
  for(let i=0;i<10;i++){bolt(-14+i*31.1,-19.5);bolt(-14+i*31.1,140.5);}
  for(let i=0;i<4;i++){bolt(-14,6+i*36.3);bolt(266,6+i*36.3);}

  // dimensions
  const dg=el("g",{"pointer-events":"none",fill:"#5d5d60","font-size":"3.4",
                   "font-family":"var(--mono)"},svg);
  el("line",{x1:0,y1:174,x2:W,y2:174,stroke:"#5d5d60","stroke-width":.35},dg);
  const t1=el("text",{x:W/2,y:180,"text-anchor":"middle"},dg);
  t1.textContent=`${W.toFixed(1)} mm · ${D.cols} pads`;
  el("line",{x1:266,y1:0,x2:266,y2:H,stroke:"#5d5d60","stroke-width":.35},dg);
  const t2=el("text",{x:270,y:H/2,"text-anchor":"middle",
                      transform:`rotate(-90 270 ${H/2})`},dg);
  t2.textContent=`${H.toFixed(1)} mm · ${D.rows} pads`;

  // rail
  document.getElementById("fl").textContent=f.label;
  document.getElementById("fu").textContent=f.unit;
  document.getElementById("fnote").textContent=f.note||"";
  const cb=document.getElementById("cbar"); cb.innerHTML="";
  for(let i=0;i<24;i++) el("rect",{x:0,y:115-i*5,width:12,height:5.2,c:0,
    fill:ramp(f.ramp,i/23)},cb);
  el("rect",{x:0,y:0,width:12,height:120,fill:"none",stroke:"#1d1f20",
             "stroke-width":.7,opacity:.5},cb);
  const ct=document.getElementById("cticks"); ct.innerHTML="";
  for(let i=4;i>=0;i--){const d=document.createElement("div");
    d.textContent=(f.lo+(f.hi-f.lo)*i/4).toFixed(f.dec); ct.appendChild(d);}
  const ds=document.getElementById("dist"); ds.innerHTML="";
  vals.forEach((v,i)=>{const t=(v-f.lo)/(f.hi-f.lo);
    const hh=Math.max(1.5,Math.min(1,Math.max(0,t))*52);
    el("rect",{x:1+i*(218/Math.max(vals.length,1)),y:56-hh,
               width:Math.max(1.6,218/Math.max(vals.length,1)-0.5),height:hh,
               fill:ramp(f.ramp,t)},ds);});
  el("line",{x1:0,y1:56,x2:220,y2:56,stroke:"#1d1f20","stroke-width":.5,opacity:.4},ds);
  document.getElementById("spread").textContent=
    `${f.lo.toFixed(f.dec)} … ${f.hi.toFixed(f.dec)} ${f.unit}`;
  document.getElementById("cap").textContent = S.side==="cathode"
    ? "cathode side · air enters bottom right, traverses right → left, leaves top left"
    : "anode side · H₂ enters bottom left, traverses left → right, leaves top right";
  document.getElementById("src").textContent =
    `${D.segments.length} segments · ${D.area.toFixed(2)} cm² · numbering: ${S.wired?"as wired":"top-left down"}`;
  document.getElementById("meta").innerHTML =
    `${D.cols} × ${D.rows} pads<br>${D.w.toFixed(1)} × ${D.h.toFixed(1)} mm<br>${D.fields.length} field(s)`;
  const pl=document.getElementById("portlist"); pl.innerHTML="";
  D.ports.forEach(p=>{const r=document.createElement("div"); r.className="rowk";
    const corner={tl:"top left",tr:"top right",bl:"bottom left",br:"bottom right"}[p.corner];
    r.innerHTML=`<span>${p.label}</span><span style="color:${p.color}">${corner}</span>`;
    pl.appendChild(r);});
  inspector();
}

function inspector(){
  const f=F(), id=S.pinned??S.hover;
  const s=D.segments.find(x=>x.wired===id);
  document.getElementById("it").textContent=s?num(s):"—";
  document.getElementById("is").textContent=s
    ? (S.wired?`top-left-down #${s.topleft}`:`as-wired #${s.wired}`)+(S.pinned?" · pinned":"")
    : "hover the plate";
  const host=document.getElementById("irows"); host.innerHTML="";
  const row=(k,v,hot)=>{const d=document.createElement("div"); d.className="rowk";
    d.innerHTML=`<span>${k}</span><span${hot?' style="color:var(--accent-700)"':""}>${v}</span>`;
    host.appendChild(d);};
  if(!s){row("—","no segment selected");return;}
  row("Area",`${s.area.toFixed(3)} cm² · ${s.np} pads`);
  row("Centre",`${s.cx.toFixed(1)}, ${s.cy.toFixed(1)} mm`);
  D.fields.forEach(x=>{const v=x.values[s.wired];
    row(x.label, v==null?"—":`${v.toFixed(x.dec)} ${x.unit}`.trim(), x.key===f.key);});
}
draw();
</script></body></html>
"""


if __name__ == "__main__":
    from plate_model import synthetic_fields

    v = synthetic_fields(150.0)
    spec = [
        ("j", "Current density", "A/cm²", "inferno", 3,
         "Highest where both reactants are fresh and the membrane is wet."),
        ("T", "Temperature", "°C", "thermal", 1,
         "Rises along both gas paths and tracks the local current."),
        ("rh", "Relative humidity", "%", "humid", 0,
         "Air enters dry at the bottom right and picks up product water."),
        ("hfr", "HFR (Rs)", "mΩ·cm²", "viridis", 1,
         "Membrane resistance — the mirror image of hydration."),
        ("rp", "Rp (charge transfer)", "mΩ·cm²", "magma", 0,
         "Grows where oxygen is depleted and liquid water blocks the GDL."),
        ("lam", "O₂ stoichiometry", "–", "cividis", 2,
         "Below about 1.2 the segment is on the edge of starvation."),
        ("sat", "Water saturation", "–", "water", 2,
         "Cold, humid, downstream regions hold the most liquid."),
    ]
    fields = [Field(k, lab, u, r, d, v[k], note) for k, lab, u, r, d, note in spec]
    out = write_html(fields, "plate.html", title="Segmented plate — 150 A",
                     subtitle="synthetic demonstration field — replace with pipeline output")
    print("wrote", out.resolve())
