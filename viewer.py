"""Web viewer for parsed results under the output folder.

Browse output/<file>/page_NNN results interactively: select a file in the
sidebar, see its pages and layout regions, click a region to inspect its
info, cropped image and vectorized polylines.

Usage:
    python viewer.py                 # serve output dir from config.json (default: output/)
    python viewer.py -o output       # serve a specific output dir
    python viewer.py --port 8000
"""
import argparse
import json
import webbrowser
from pathlib import Path
from threading import Timer

from flask import Flask, Response, abort, jsonify, send_file

ROOT = Path(__file__).parent


# ---------------------------------------------------------------- server ---

def create_app(out_root: Path) -> Flask:
    app = Flask(__name__)
    out_root = out_root.resolve()

    def safe_path(rel: str) -> Path:
        p = (out_root / rel).resolve()
        if not p.is_relative_to(out_root):
            abort(403)
        if not p.exists():
            abort(404)
        return p

    def read_json(p: Path):
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    @app.get("/")
    def index() -> Response:
        return Response(PAGE, mimetype="text/html")

    @app.get("/api/files")
    def api_files():
        """All parsed files: directories under out_root that contain result.json."""
        items = []
        if out_root.exists():
            for d in sorted(out_root.iterdir()):
                rj = d / "result.json"
                if d.is_dir() and rj.exists():
                    try:
                        items.append({"name": d.name, **read_json(rj)})
                    except (json.JSONDecodeError, OSError):
                        items.append({"name": d.name, "error": "result.json unreadable"})
        return jsonify({"output_dir": str(out_root), "files": items})

    @app.get("/api/layout/<name>/<page_dir>")
    def api_layout(name: str, page_dir: str):
        p = safe_path(f"{name}/{page_dir}/layout.json")
        data = read_json(p)
        data["has_overlay"] = (p.parent / "overlay.png").exists()
        data["has_page_image"] = (p.parent / "page.png").exists()
        data["has_native_vectors"] = (p.parent / "native_vectors.json").exists()
        return jsonify(data)

    @app.get("/api/vectors/<name>/<page_dir>/<rid>")
    def api_vectors(name: str, page_dir: str, rid: str):
        return jsonify(read_json(safe_path(f"{name}/{page_dir}/vectors/{rid}.json")))

    @app.get("/api/native/<name>/<page_dir>")
    def api_native(name: str, page_dir: str):
        return jsonify(read_json(safe_path(f"{name}/{page_dir}/native_vectors.json")))

    @app.get("/files/<path:rel>")
    def files(rel: str):
        return send_file(safe_path(rel))

    return app


# -------------------------------------------------------------- frontend ---

PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Doc Layout Parser Viewer</title>
<style>
:root{
  --bg:#0e1015; --panel:#151821; --panel2:#1a1e2a; --border:#262c3b;
  --fg:#dfe4ee; --fg-dim:#8a93a8; --accent:#4f8cff; --accent-dim:#2a3f66;
  --c-text:#3b9dff; --c-dimension:#ff5252; --c-annotation:#ffa726;
  --c-drawing:#2ecc71; --c-image:#d05ce3;
  --radius:10px; --font:13px/1.5 "Segoe UI","Malgun Gothic",system-ui,sans-serif;
}
*{box-sizing:border-box; margin:0}
html,body{height:100%}
body{font:var(--font); background:var(--bg); color:var(--fg); overflow:hidden}
#app{display:grid; grid-template-columns:var(--w-side,250px) 5px minmax(0,1fr) 5px var(--w-detail,340px);
  grid-template-rows:100vh}

/* ---------- splitters ---------- */
.vsplit{background:var(--border); cursor:col-resize; position:relative; z-index:5; transition:background .12s}
.hsplit{background:var(--border); cursor:row-resize; height:5px; flex:none; position:relative; z-index:5;
  transition:background .12s; display:none}
.vsplit:hover,.hsplit:hover,.vsplit.drag,.hsplit.drag{background:var(--accent)}
.vsplit::after{content:""; position:absolute; left:-3px; right:-3px; top:0; bottom:0}
.hsplit::after{content:""; position:absolute; top:-3px; bottom:-3px; left:0; right:0}
body.resizing{cursor:col-resize; user-select:none}
body.resizing-v{cursor:row-resize; user-select:none}
body.resizing iframe, body.resizing img, body.resizing-v img{pointer-events:none}

/* ---------- sidebar ---------- */
#side{background:var(--panel); border-right:1px solid var(--border); display:flex; flex-direction:column; min-width:0}
#side h1{font-size:14px; font-weight:600; padding:14px 16px 4px; letter-spacing:.2px}
#side h1 span{color:var(--accent)}
#side .sub{padding:0 16px 10px; color:var(--fg-dim); font-size:11px; border-bottom:1px solid var(--border);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
#fileList{overflow-y:auto; flex:1; padding:8px}
.fitem{border-radius:var(--radius); margin-bottom:2px}
.fitem>.fhead{display:flex; align-items:center; gap:8px; padding:7px 10px; cursor:pointer; border-radius:var(--radius)}
.fitem>.fhead:hover{background:var(--panel2)}
.fitem.open>.fhead{background:var(--panel2)}
.fhead .name{flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:500}
.fhead .cnt{color:var(--fg-dim); font-size:11px}
.fhead .arrow{color:var(--fg-dim); font-size:10px; transition:transform .15s}
.fitem.open .arrow{transform:rotate(90deg)}
.pages{display:none; padding:2px 0 6px}
.fitem.open .pages{display:block}
.pitem{display:flex; align-items:center; gap:8px; padding:5px 10px 5px 28px; cursor:pointer;
  border-radius:8px; color:var(--fg-dim); font-size:12px}
.pitem:hover{background:var(--panel2); color:var(--fg)}
.pitem.sel{background:var(--accent-dim); color:var(--fg)}
.pitem .rc{margin-left:auto; font-size:10px; color:var(--fg-dim)}

/* ---------- main ---------- */
#main{display:flex; flex-direction:column; min-width:0; background:var(--bg)}
#toolbar{display:flex; align-items:center; gap:10px; flex-wrap:wrap; padding:8px 14px;
  background:var(--panel); border-bottom:1px solid var(--border)}
#crumb{font-size:12px; color:var(--fg-dim); margin-right:auto; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
#crumb b{color:var(--fg); font-weight:600}
.tgroup{display:flex; align-items:center; gap:4px; background:var(--panel2); border:1px solid var(--border);
  border-radius:8px; padding:3px}
.tbtn{border:0; background:transparent; color:var(--fg-dim); font:inherit; font-size:12px;
  padding:3px 10px; border-radius:6px; cursor:pointer; white-space:nowrap}
.tbtn:hover{color:var(--fg)}
.tbtn.on{background:var(--accent-dim); color:var(--fg)}
.tbtn:disabled{opacity:.35; cursor:default}
.chip{display:inline-flex; align-items:center; gap:5px; border:1px solid var(--border); background:var(--panel2);
  color:var(--fg-dim); border-radius:999px; padding:3px 10px; font-size:11.5px; cursor:pointer; user-select:none}
.chip .dot{width:8px; height:8px; border-radius:50%}
.chip.on{color:var(--fg); border-color:var(--fg-dim)}
.chip:not(.on){opacity:.45}
#canvasWrap{flex:1; position:relative; overflow:hidden; cursor:grab; background:
  radial-gradient(circle at 50% 40%, #141824 0%, var(--bg) 70%)}
#canvasWrap.panning{cursor:grabbing}
#stage{position:absolute; transform-origin:0 0}
#pageImg{display:block; user-select:none; -webkit-user-drag:none;
  box-shadow:0 8px 40px rgba(0,0,0,.55); background:#fff}
#ov{position:absolute; left:0; top:0; overflow:visible}
#ov .rgn rect{fill:transparent; stroke-width:2; vector-effect:non-scaling-stroke; cursor:pointer}
#ov .rgn:hover rect{fill:rgba(255,255,255,.07)}
#ov .rgn.sel rect{stroke-width:3.5; fill:rgba(79,140,255,.10)}
#ov .rgn text{font:600 13px sans-serif; paint-order:stroke; stroke:#000a; stroke-width:3px; pointer-events:none}
#ov .vec path{fill:none; stroke-width:1.2; vector-effect:non-scaling-stroke; pointer-events:none}
#empty{position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  color:var(--fg-dim); font-size:14px; flex-direction:column; gap:8px}
#hud{position:absolute; right:12px; bottom:12px; background:var(--panel); border:1px solid var(--border);
  border-radius:8px; padding:4px 10px; font-size:11px; color:var(--fg-dim)}

/* ---------- detail ---------- */
#detail{background:var(--panel); border-left:1px solid var(--border); display:flex; flex-direction:column; min-width:0}
#dhead{padding:10px 14px; border-bottom:1px solid var(--border); font-weight:600; font-size:13px;
  display:flex; align-items:center; gap:8px}
#dhead .n{margin-left:auto; font-weight:400; color:var(--fg-dim); font-size:11px}
#rlist{overflow-y:auto; flex:1; min-height:60px; padding:6px}
.rrow{display:flex; align-items:center; gap:8px; padding:6px 8px; border-radius:8px; cursor:pointer; font-size:12px}
.rrow:hover{background:var(--panel2)}
.rrow.sel{background:var(--accent-dim)}
.rrow .dot{width:9px; height:9px; border-radius:3px; flex:none}
.rrow .id{font-family:Consolas,monospace; color:var(--fg-dim); flex:none}
.rrow .snip{flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--fg-dim)}
.rrow .cf{flex:none; font-size:10.5px; color:var(--fg-dim)}
#rdetail{overflow-y:auto; flex:none; height:var(--h-detail,45%); min-height:100px; display:none; background:var(--panel)}
#rdetail.show{display:block}
#detail.has-detail #splitD{display:block}
#rdetail .inner{padding:12px 14px 16px}
#rdetail h3{font-size:13px; display:flex; align-items:center; gap:8px; margin-bottom:8px}
#rdetail h3 .badge{font-size:10.5px; padding:2px 8px; border-radius:999px; color:#fff; font-weight:600}
#rdetail h3 .zoombtn{margin-left:auto}
#rdetail table{width:100%; border-collapse:collapse; font-size:11.5px; margin-bottom:10px}
#rdetail td{padding:3px 0; vertical-align:top}
#rdetail td:first-child{color:var(--fg-dim); width:88px; white-space:nowrap}
#rdetail .txtbox{background:var(--panel2); border:1px solid var(--border); border-radius:8px;
  padding:8px 10px; font-size:12.5px; margin-bottom:10px; word-break:break-all; white-space:pre-wrap}
#rdetail .imgbox{background:#fff; border:1px solid var(--border); border-radius:8px; overflow:hidden; margin-bottom:10px}
#rdetail .imgbox img, #rdetail .imgbox svg{display:block; width:100%; height:auto; max-height:260px; object-fit:contain}
#rdetail .cap{font-size:10.5px; color:var(--fg-dim); margin:-6px 0 10px 2px}
.minitabs{display:flex; gap:4px; margin-bottom:8px}
.smallbtn{border:1px solid var(--border); background:var(--panel2); color:var(--fg-dim); font:inherit;
  font-size:11px; padding:3px 10px; border-radius:7px; cursor:pointer}
.smallbtn:hover{color:var(--fg)}
.smallbtn.on{background:var(--accent-dim); color:var(--fg); border-color:var(--accent-dim)}
.spin{color:var(--fg-dim); font-size:11.5px; padding:6px 2px}
::-webkit-scrollbar{width:10px; height:10px}
::-webkit-scrollbar-thumb{background:#2a3040; border-radius:5px; border:2px solid var(--panel)}
::-webkit-scrollbar-track{background:transparent}
</style>
</head>
<body>
<div id="app">
  <aside id="side">
    <h1>Doc Layout Parser <span>Viewer</span></h1>
    <div class="sub" id="outdir"></div>
    <div id="fileList"></div>
  </aside>

  <div class="vsplit" id="splitL" title="드래그로 크기 조절, 더블클릭으로 초기화"></div>

  <section id="main">
    <div id="toolbar">
      <div id="crumb">파일을 선택하세요</div>
      <div class="tgroup" id="baseGroup">
        <button class="tbtn on" data-base="page">원본</button>
        <button class="tbtn" data-base="overlay">오버레이</button>
      </div>
      <div class="tgroup">
        <button class="tbtn on" id="lyBbox">영역 박스</button>
        <button class="tbtn" id="lyVec">벡터</button>
        <button class="tbtn" id="lyNative" style="display:none">PDF 벡터</button>
      </div>
      <div id="typeChips" style="display:flex; gap:5px"></div>
      <div class="tgroup">
        <button class="tbtn" id="zoomFit" title="화면 맞춤">맞춤</button>
        <button class="tbtn" id="zoom100" title="100%">1:1</button>
      </div>
    </div>
    <div id="canvasWrap">
      <div id="stage">
        <img id="pageImg" alt="">
        <svg id="ov"><g class="vec" id="vecLayer"></g><g class="vec" id="nativeLayer"></g><g id="rgnLayer"></g></svg>
      </div>
      <div id="empty"><div style="font-size:32px">📐</div><div>왼쪽에서 파일과 페이지를 선택하세요</div></div>
      <div id="hud" style="display:none"></div>
    </div>
  </section>

  <div class="vsplit" id="splitR" title="드래그로 크기 조절, 더블클릭으로 초기화"></div>

  <aside id="detail">
    <div id="dhead">레이아웃 영역 <span class="n" id="rcount"></span></div>
    <div id="rlist"></div>
    <div class="hsplit" id="splitD" title="드래그로 크기 조절, 더블클릭으로 초기화"></div>
    <div id="rdetail"><div class="inner" id="rdetailInner"></div></div>
  </aside>
</div>

<script>
"use strict";
const TYPE_COLORS = {text:"#3b9dff", dimension:"#ff5252", annotation:"#ffa726", drawing:"#2ecc71", image:"#d05ce3"};
const TYPE_LABELS = {text:"텍스트", dimension:"치수", annotation:"주석", drawing:"도면", image:"이미지"};
const GROUP_PALETTE = ["#2ecc71","#3b9dff","#ff8f3d","#d05ce3","#00c2c7","#ffd23b","#ff5252","#9fd63b",
                       "#7a7cff","#ff7ab8","#5bd0ff","#c0a06a"];
const $ = s => document.querySelector(s);
const el = (tag, attrs={}, html) => { const e=document.createElement(tag);
  for(const [k,v] of Object.entries(attrs)) e.setAttribute(k,v); if(html!==undefined) e.innerHTML=html; return e; };
const svgEl = (tag, attrs={}) => { const e=document.createElementNS("http://www.w3.org/2000/svg",tag);
  for(const [k,v] of Object.entries(attrs)) e.setAttribute(k,v); return e; };
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const enc = encodeURIComponent;

const state = {
  files: [], file: null, page: null, layout: null, selId: null,
  base: "page", layers: {bbox:true, vec:false, native:false},
  types: {text:true, dimension:true, annotation:true, drawing:true, image:true},
  view: {x:0, y:0, k:1},
  vecCache: {}, nativeCache: {}, detailTab: "crop",
};

async function jget(url){ const r = await fetch(url); if(!r.ok) throw new Error(url+" -> "+r.status); return r.json(); }
const fileUrl = rel => "/files/" + rel.split("/").map(enc).join("/");

/* ---------------- sidebar ---------------- */
async function loadFiles(){
  const data = await jget("/api/files");
  state.files = data.files;
  $("#outdir").textContent = data.output_dir;
  $("#outdir").title = data.output_dir;
  const list = $("#fileList"); list.innerHTML = "";
  if(!state.files.length){
    list.appendChild(el("div", {style:"padding:14px;color:var(--fg-dim);font-size:12px"},
      "파싱 결과가 없습니다.<br>먼저 <code>python main.py</code>를 실행하세요."));
    return;
  }
  for(const f of state.files){
    const item = el("div", {class:"fitem"});
    const head = el("div", {class:"fhead"},
      `<span class="arrow">▶</span><span class="name" title="${esc(f.source_file||f.name)}">${esc(f.name)}</span>
       <span class="cnt">${f.num_pages ?? "?"}p</span>`);
    const pages = el("div", {class:"pages"});
    for(const p of (f.pages || [])){
      const total = p.num_regions ?? 0;
      const row = el("div", {class:"pitem", "data-file":f.name, "data-page":p.dir},
        `<span>페이지 ${p.page}</span><span class="rc">${total} 영역</span>`);
      row.onclick = () => selectPage(f.name, p.dir);
      pages.appendChild(row);
    }
    head.onclick = () => {
      const wasOpen = item.classList.contains("open");
      item.classList.toggle("open");
      if(!wasOpen && f.pages && f.pages.length) selectPage(f.name, f.pages[0].dir);
    };
    item.append(head, pages);
    list.appendChild(item);
  }
  // auto-open the first file
  const first = state.files.find(f => f.pages && f.pages.length);
  if(first){
    list.querySelector(".fitem").classList.add("open");
    selectPage(first.name, first.pages[0].dir);
  }
}

/* ---------------- page load ---------------- */
async function selectPage(file, pageDir){
  state.file = file; state.page = pageDir; state.selId = null;
  state.vecCache = {}; state.nativeCache = {};
  document.querySelectorAll(".pitem").forEach(x =>
    x.classList.toggle("sel", x.dataset.file===file && x.dataset.page===pageDir));

  const layout = await jget(`/api/layout/${enc(file)}/${enc(pageDir)}`);
  state.layout = layout;
  $("#empty").style.display = "none";
  $("#hud").style.display = "";
  $("#crumb").innerHTML = `<b>${esc(file)}</b> / ${esc(pageDir)} · ${layout.size.width}×${layout.size.height}px` +
    (layout.scale && layout.scale !== 1 ? ` (scale ${layout.scale}×)` : "");

  // base image toggle availability
  const obtn = document.querySelector('[data-base="overlay"]');
  obtn.disabled = !layout.has_overlay;
  if(!layout.has_overlay && state.base === "overlay") setBase("page");
  $("#lyNative").style.display = layout.has_native_vectors ? "" : "none";
  if(!layout.has_native_vectors) state.layers.native = false;

  const img = $("#pageImg");
  img.onload = () => { sizeOverlay(); fitView(); };
  img.src = baseImageUrl();
  buildTypeChips();
  renderRegions();
  renderList();
  closeDetail();
  refreshLayerButtons();
  if(state.layers.vec) loadPageVectors();
  if(state.layers.native) loadNativeVectors();
}

function baseImageUrl(){
  return fileUrl(`${state.file}/${state.page}/${state.base === "overlay" ? "overlay.png" : "page.png"}`);
}
function setBase(b){
  state.base = b;
  document.querySelectorAll("#baseGroup .tbtn").forEach(x => x.classList.toggle("on", x.dataset.base===b));
  if(state.layout){ const img=$("#pageImg"); img.onload=()=>sizeOverlay(); img.src = baseImageUrl(); }
}

function sizeOverlay(){
  const img = $("#pageImg"), ov = $("#ov");
  ov.setAttribute("width", img.naturalWidth); ov.setAttribute("height", img.naturalHeight);
  ov.setAttribute("viewBox", `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
}

/* ---------------- overlay regions ---------------- */
function renderRegions(){
  const layer = $("#rgnLayer"); layer.innerHTML = "";
  if(!state.layout || !state.layers.bbox) return;
  for(const r of state.layout.regions){
    if(!state.types[r.type]) continue;
    const [x0,y0,x1,y1] = r.bbox;
    const color = TYPE_COLORS[r.type] || "#999";
    const g = svgEl("g", {class:"rgn" + (r.id===state.selId ? " sel" : ""), "data-id":r.id});
    g.appendChild(svgEl("rect", {x:x0, y:y0, width:x1-x0, height:y1-y0, stroke:color}));
    const t = svgEl("text", {x:x0+3, y:Math.max(14, y0-5), fill:color});
    t.textContent = `${r.id} ${r.type}`;
    g.appendChild(t);
    g.addEventListener("click", ev => { ev.stopPropagation(); selectRegion(r.id); });
    layer.appendChild(g);
  }
}

function polylinesToPath(polys){
  let d = "";
  for(const pl of polys){
    const pts = pl.points;
    if(!pts || pts.length < 2) continue;
    d += `M${pts[0][0]} ${pts[0][1]}`;
    for(let i=1;i<pts.length;i++) d += `L${pts[i][0]} ${pts[i][1]}`;
    if(pl.closed) d += "Z";
  }
  return d;
}

async function loadPageVectors(){
  const layer = $("#vecLayer"); layer.innerHTML = "";
  if(!state.layout) return;
  const file = state.file, page = state.page;
  const regions = state.layout.regions.filter(r => r.vector_file);
  for(const r of regions){
    // abort if the layer was toggled off or the page changed while loading
    if(!state.layers.vec || state.file !== file || state.page !== page) return;
    try{
      const v = await getVectors(r.id);
      if(!state.layers.vec || state.file !== file || state.page !== page || !v) return;
      layer.appendChild(svgEl("path", {d: polylinesToPath(v.polylines), stroke: "#2ecc71"}));
    }catch(e){ console.warn("vectors failed", r.id, e); }
  }
}
async function getVectors(rid){
  const key = `${state.file}/${state.page}/${rid}`;
  if(!(key in state.vecCache))
    state.vecCache[key] = await jget(`/api/vectors/${enc(state.file)}/${enc(state.page)}/${enc(rid)}`);
  return state.vecCache[key];
}
async function loadNativeVectors(){
  const layer = $("#nativeLayer"); layer.innerHTML = "";
  if(!state.layout || !state.layout.has_native_vectors) return;
  const file = state.file, page = state.page, key = `${file}/${page}`;
  try{
    if(!(key in state.nativeCache))
      state.nativeCache[key] = await jget(`/api/native/${enc(file)}/${enc(page)}`);
    if(!state.layers.native || state.file !== file || state.page !== page) return;
    layer.appendChild(svgEl("path", {d: polylinesToPath(state.nativeCache[key].polylines), stroke:"#c94fd8"}));
  }catch(e){ console.warn("native vectors failed", e); }
}

/* ---------------- type chips / layer buttons ---------------- */
function buildTypeChips(){
  const box = $("#typeChips"); box.innerHTML = "";
  const counts = {};
  for(const r of (state.layout?.regions || [])) counts[r.type] = (counts[r.type]||0)+1;
  for(const t of Object.keys(TYPE_COLORS)){
    if(!(t in counts)) continue;
    const chip = el("span", {class:"chip" + (state.types[t] ? " on" : "")},
      `<span class="dot" style="background:${TYPE_COLORS[t]}"></span>${TYPE_LABELS[t]} ${counts[t]}`);
    chip.onclick = () => { state.types[t] = !state.types[t]; buildTypeChips(); renderRegions(); renderList(); };
    box.appendChild(chip);
  }
}
function refreshLayerButtons(){
  $("#lyBbox").classList.toggle("on", state.layers.bbox);
  $("#lyVec").classList.toggle("on", state.layers.vec);
  $("#lyNative").classList.toggle("on", state.layers.native);
}
$("#lyBbox").onclick = () => { state.layers.bbox = !state.layers.bbox; refreshLayerButtons(); renderRegions(); };
$("#lyVec").onclick = () => {
  state.layers.vec = !state.layers.vec; refreshLayerButtons();
  if(state.layers.vec) loadPageVectors(); else $("#vecLayer").innerHTML = "";
};
$("#lyNative").onclick = () => {
  state.layers.native = !state.layers.native; refreshLayerButtons();
  if(state.layers.native) loadNativeVectors(); else $("#nativeLayer").innerHTML = "";
};
document.querySelectorAll("#baseGroup .tbtn").forEach(b => b.onclick = () => setBase(b.dataset.base));

/* ---------------- region list ---------------- */
function snippet(r){
  if(r.text) return r.text;
  if(r.type === "drawing") return `폴리라인 ${r.num_polylines ?? 0}개`;
  if(r.type === "image") return "래스터 이미지";
  return "";
}
function renderList(){
  const list = $("#rlist"); list.innerHTML = "";
  const regions = (state.layout?.regions || []).filter(r => state.types[r.type]);
  $("#rcount").textContent = state.layout ? `${regions.length}/${state.layout.num_regions}` : "";
  for(const r of regions){
    const row = el("div", {class:"rrow" + (r.id===state.selId ? " sel":""), "data-id":r.id},
      `<span class="dot" style="background:${TYPE_COLORS[r.type]||"#999"}"></span>
       <span class="id">${r.id}</span>
       <span class="snip" title="${esc(snippet(r))}">${esc(snippet(r))}</span>
       <span class="cf">${Math.round((r.confidence||0)*100)}%</span>`);
    row.onclick = () => selectRegion(r.id);
    list.appendChild(row);
  }
}

/* ---------------- selection & detail ---------------- */
function selectRegion(id){
  state.selId = id;
  document.querySelectorAll("#rgnLayer .rgn").forEach(g => g.classList.toggle("sel", g.dataset.id===id));
  document.querySelectorAll("#rlist .rrow").forEach(x => x.classList.toggle("sel", x.dataset.id===id));
  const row = document.querySelector(`#rlist .rrow[data-id="${id}"]`);
  if(row) row.scrollIntoView({block:"nearest"});
  renderDetail();
}
function closeDetail(){ $("#rdetail").classList.remove("show"); $("#detail").classList.remove("has-detail"); }

async function renderDetail(){
  const r = state.layout?.regions.find(x => x.id === state.selId);
  const panel = $("#rdetail"), inner = $("#rdetailInner");
  if(!r){ closeDetail(); return; }
  panel.classList.add("show");
  $("#detail").classList.add("has-detail");
  const color = TYPE_COLORS[r.type] || "#999";
  const [x0,y0,x1,y1] = r.bbox.map(v => Math.round(v));
  let html = `<h3><span class="badge" style="background:${color}">${TYPE_LABELS[r.type]||r.type}</span>
      <span>${r.id}</span>
      <button class="smallbtn zoombtn" onclick="zoomToRegion('${r.id}')">영역으로 확대</button>
      <button class="smallbtn" onclick="closeDetail()">✕</button></h3>
    <table>
      <tr><td>bbox</td><td>[${x0}, ${y0}] – [${x1}, ${y1}] &nbsp;(${x1-x0}×${y1-y0}px)</td></tr>
      <tr><td>신뢰도</td><td>${(r.confidence ?? 0).toFixed(3)}</td></tr>
      <tr><td>분류 방법</td><td>${esc(r.source || "-")}</td></tr>`;
  if(r.metrics) html += `<tr><td>metrics</td><td>${Object.entries(r.metrics)
      .map(([k,v])=>`${k}=${v}`).join(", ")}</td></tr>`;
  if(r.words?.length) html += `<tr><td>단어 수</td><td>${r.words.length}</td></tr>`;
  if(r.num_polylines) html += `<tr><td>폴리라인</td><td>${r.num_polylines}개</td></tr>`;
  html += `</table>`;
  if(r.text) html += `<div class="txtbox">${esc(r.text)}</div>`;

  const hasCrop = !!r.image_file, hasVec = !!r.vector_file;
  if(hasCrop || hasVec){
    if(state.detailTab === "vec" && !hasVec) state.detailTab = "crop";
    if(state.detailTab === "crop" && !hasCrop) state.detailTab = "vec";
    html += `<div class="minitabs">`;
    if(hasCrop) html += `<button class="smallbtn ${state.detailTab==="crop"?"on":""}" onclick="setDetailTab('crop')">이미지</button>`;
    if(hasVec)  html += `<button class="smallbtn ${state.detailTab==="vec"?"on":""}" onclick="setDetailTab('vec')">벡터</button>`;
    if(r.svg_file) html += `<a class="smallbtn" style="text-decoration:none"
        href="${fileUrl(state.file+"/"+state.page+"/"+r.svg_file)}" target="_blank">SVG 열기</a>`;
    html += `</div><div id="mediaBox"></div>`;
  }
  inner.innerHTML = html;

  const box = $("#mediaBox");
  if(!box) return;
  if(state.detailTab === "crop" && hasCrop){
    box.innerHTML = `<div class="imgbox"><img src="${fileUrl(state.file+"/"+state.page+"/"+r.image_file)}"></div>
                     <div class="cap">영역 crop 이미지</div>`;
  }else if(state.detailTab === "vec" && hasVec){
    box.innerHTML = `<div class="spin">벡터 로딩중…</div>`;
    try{
      const v = await getVectors(r.id);
      if(state.selId !== r.id) return;   // selection changed while loading
      const [bx0,by0,bx1,by1] = v.bbox;
      const svg = svgEl("svg", {viewBox:`${bx0} ${by0} ${bx1-bx0} ${by1-by0}`,
                                xmlns:"http://www.w3.org/2000/svg"});
      const byGroup = {};
      for(const pl of v.polylines) (byGroup[pl.group] ??= []).push(pl);
      for(const [grp, pls] of Object.entries(byGroup)){
        svg.appendChild(svgEl("path", {d: polylinesToPath(pls), fill:"none",
          stroke: GROUP_PALETTE[grp % GROUP_PALETTE.length],
          "stroke-width":"1", "vector-effect":"non-scaling-stroke"}));
      }
      box.innerHTML = "";
      const wrap = el("div", {class:"imgbox"}); wrap.appendChild(svg);
      box.append(wrap, el("div", {class:"cap"},
        `폴리라인 ${v.num_polylines}개 · 연결그룹 ${v.num_groups}개 (그룹별 색상)`));
    }catch(e){ box.innerHTML = `<div class="spin">벡터 로드 실패: ${esc(e.message)}</div>`; }
  }
}
function setDetailTab(t){ state.detailTab = t; renderDetail(); }

/* ---------------- pan & zoom ---------------- */
const wrap = $("#canvasWrap"), stage = $("#stage");
function applyView(){
  const v = state.view;
  stage.style.transform = `translate(${v.x}px, ${v.y}px) scale(${v.k})`;
  $("#hud").textContent = `${Math.round(v.k*100)}%`;
}
function fitView(){
  const img = $("#pageImg");
  if(!img.naturalWidth) return;
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  const k = Math.min(cw/img.naturalWidth, ch/img.naturalHeight) * 0.94;
  state.view = {k, x:(cw - img.naturalWidth*k)/2, y:(ch - img.naturalHeight*k)/2};
  applyView();
}
function zoomToRegion(id){
  const r = state.layout?.regions.find(x => x.id === id);
  if(!r) return;
  const [x0,y0,x1,y1] = r.bbox;
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  const k = Math.min(cw/(x1-x0), ch/(y1-y0)) * 0.8;
  const kk = Math.min(Math.max(k, 0.02), 30);
  state.view = {k:kk, x: cw/2 - (x0+x1)/2*kk, y: ch/2 - (y0+y1)/2*kk};
  applyView();
}
wrap.addEventListener("wheel", ev => {
  ev.preventDefault();
  const v = state.view;
  const factor = Math.exp(-ev.deltaY * 0.0012);
  const k2 = Math.min(Math.max(v.k * factor, 0.02), 30);
  const rect = wrap.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  v.x = mx - (mx - v.x) * (k2 / v.k);
  v.y = my - (my - v.y) * (k2 / v.k);
  v.k = k2;
  applyView();
}, {passive:false});
let pan = null;
wrap.addEventListener("pointerdown", ev => {
  if(ev.button !== 0) return;
  pan = {sx:ev.clientX, sy:ev.clientY, ox:state.view.x, oy:state.view.y, moved:false};
  wrap.classList.add("panning"); wrap.setPointerCapture(ev.pointerId);
});
wrap.addEventListener("pointermove", ev => {
  if(!pan) return;
  const dx = ev.clientX - pan.sx, dy = ev.clientY - pan.sy;
  if(Math.abs(dx) + Math.abs(dy) > 3) pan.moved = true;
  state.view.x = pan.ox + dx; state.view.y = pan.oy + dy; applyView();
});
wrap.addEventListener("pointerup", ev => {
  wrap.classList.remove("panning");
  if(pan && !pan.moved) { /* plain click on background: keep selection */ }
  pan = null;
});
$("#zoomFit").onclick = fitView;
$("#zoom100").onclick = () => {
  const cw = wrap.clientWidth, ch = wrap.clientHeight, img = $("#pageImg");
  state.view = {k:1, x:(cw-img.naturalWidth)/2, y:(ch-img.naturalHeight)/2}; applyView();
};
window.addEventListener("resize", () => { if(state.layout) fitView(); });

/* ---------------- panel splitters ---------------- */
const rootStyle = document.documentElement.style;
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
function loadSplit(){ try{ return JSON.parse(localStorage.getItem("viewerSplit") || "{}"); }catch(e){ return {}; } }
function saveSplit(k, v){
  try{ const s = loadSplit(); s[k] = v; localStorage.setItem("viewerSplit", JSON.stringify(s)); }catch(e){}
}
function initSplitter(id, onMove, onReset, vertical){
  const bar = document.getElementById(id);
  bar.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    bar.classList.add("drag");
    document.body.classList.add(vertical ? "resizing-v" : "resizing");
    bar.setPointerCapture(ev.pointerId);
    const move = e => onMove(e);
    const up = () => {
      bar.classList.remove("drag");
      document.body.classList.remove("resizing", "resizing-v");
      bar.removeEventListener("pointermove", move);
      bar.removeEventListener("pointerup", up);
    };
    bar.addEventListener("pointermove", move);
    bar.addEventListener("pointerup", up);
  });
  bar.addEventListener("dblclick", onReset);
}
initSplitter("splitL", e => {
  const w = clamp(e.clientX, 150, Math.min(560, window.innerWidth * 0.4));
  rootStyle.setProperty("--w-side", w + "px"); saveSplit("side", Math.round(w));
}, () => { rootStyle.removeProperty("--w-side"); saveSplit("side", null); });
initSplitter("splitR", e => {
  const w = clamp(window.innerWidth - e.clientX, 220, Math.min(760, window.innerWidth * 0.6));
  rootStyle.setProperty("--w-detail", w + "px"); saveSplit("detail", Math.round(w));
}, () => { rootStyle.removeProperty("--w-detail"); saveSplit("detail", null); });
initSplitter("splitD", e => {
  const panel = $("#detail").getBoundingClientRect();
  const pct = clamp((panel.bottom - e.clientY) / panel.height * 100, 15, 85);
  rootStyle.setProperty("--h-detail", pct.toFixed(1) + "%"); saveSplit("hdetail", +pct.toFixed(1));
}, () => { rootStyle.removeProperty("--h-detail"); saveSplit("hdetail", null); }, true);
(function restoreSplit(){
  const s = loadSplit();
  if(s.side)    rootStyle.setProperty("--w-side", s.side + "px");
  if(s.detail)  rootStyle.setProperty("--w-detail", s.detail + "px");
  if(s.hdetail) rootStyle.setProperty("--h-detail", s.hdetail + "%");
})();

loadFiles().catch(e => { $("#fileList").innerHTML =
  `<div style="padding:14px;color:#ff5252;font-size:12px">로드 실패: ${esc(e.message)}</div>`; });
</script>
</body>
</html>
"""


# ------------------------------------------------------------------ main ---

def main():
    ap = argparse.ArgumentParser(description="Web viewer for doc_layout_parser parsing results")
    ap.add_argument("-o", "--output", default=None,
                    help="output folder to browse (default: output_dir from config.json)")
    ap.add_argument("-c", "--config", default=str(ROOT / "config.json"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open the web browser")
    args = ap.parse_args()

    if args.output:
        out_root = Path(args.output)
    else:
        out_dir = "output"
        cfg_path = Path(args.config)
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                out_dir = json.load(f).get("output_dir", "output")
        out_root = Path(out_dir) if Path(out_dir).is_absolute() else ROOT / out_dir

    if not out_root.exists():
        print(f"[WARN] output folder not found: {out_root} (run main.py first)")

    app = create_app(out_root)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving {out_root} at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
