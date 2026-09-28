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
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
from pathlib import Path
from threading import Timer

import fitz
from flask import Flask, Response, abort, jsonify, request, send_file
from pipeline.config import load_config, preflight, resolve_config_path
from pipeline.loader import IMAGE_EXTS
from werkzeug.exceptions import HTTPException

ROOT = Path(__file__).parent


# ---------------------------------------------------------------- server ---


def create_app(
    out_root: Path,
    *,
    input_root: Path | None = None,
    config_path: Path | None = None,
    workspace_root: Path | None = None,
    enable_local_processing: bool = False,
) -> Flask:
    app = Flask(__name__)
    out_root = out_root.resolve()
    workspace_root = (workspace_root or Path.cwd()).resolve()
    initial_input = (input_root or (workspace_root / "input")).resolve()
    selected_config = config_path.resolve() if config_path else None
    jobs = {}
    parser_processes = {}
    cancel_requests = set()
    jobs_lock = threading.Lock()

    def local_processing_required():
        if not enable_local_processing:
            abort(404)
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            abort(403)

    def allowed_roots():
        roots = (
            workspace_root,
            Path.home(),
            initial_input,
          out_root,
            selected_config.parent if selected_config else workspace_root,
        )
        return tuple(dict.fromkeys(path.resolve() for path in roots))

    def local_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser().resolve()
        if not any(
            path == root or path.is_relative_to(root) for root in allowed_roots()
        ):
            abort(403, "Path is outside the local workspace and home folders")
        return path

    def run_parser(job_id: str, cfg: dict, markdown: bool, skip_existing: bool,
             page_counts: dict[str, int]) -> None:
        config_file = None
        output = ""
        try:
          with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
          ) as stream:
            json.dump(cfg, stream, ensure_ascii=False, indent=2)
            config_file = Path(stream.name)
          command = [sys.executable, "-u", str(ROOT / "main.py"), "--config", str(config_file)]
          if markdown:
            command.append("--markdown")
          if skip_existing:
            command.append("--skip-existing")
          child_env = os.environ.copy()
          child_env["PYTHONIOENCODING"] = "utf-8"
          process = subprocess.Popen(
            command,
            cwd=workspace_root,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
          )
          with jobs_lock:
            parser_processes[job_id] = process
            stop_requested = job_id in cancel_requests
          if stop_requested:
            process.terminate()
          suppress_warning_gap = False
          for line in process.stdout:
            stripped = line.strip()
            if stripped == "MuPDF error: syntax error: unknown keyword: '0rg'":
              suppress_warning_gap = True
              continue
            if not stripped and suppress_warning_gap:
              continue
            suppress_warning_gap = False
            output = (output + line)[-12000:]
            with jobs_lock:
              job = jobs[job_id]
              progress = job["progress"]
              if stripped.startswith("[SKIP] "):
                name = stripped[len("[SKIP] "):].split(": output already exists", 1)[0]
                progress["completed_pages"] = min(
                    progress["total_pages"],
                    progress["completed_pages"] + page_counts.get(name, 1),
                )
              elif stripped.startswith("=== ") and stripped.endswith(" ==="):
                progress["current_file"] = stripped[4:-4]
                progress["current_page"] = 0
              else:
                page_match = re.match(r"page (\d+):", stripped)
                if page_match:
                  progress["current_page"] = int(page_match.group(1))
                  progress["completed_pages"] = min(
                      progress["total_pages"], progress["completed_pages"] + 1
                  )
              job["log"] = output
          returncode = process.wait()
          with jobs_lock:
            stopped = job_id in cancel_requests
            status = "stopped" if stopped else "complete" if returncode == 0 else "failed"
            if status == "complete":
              jobs[job_id]["progress"]["completed_pages"] = jobs[job_id]["progress"]["total_pages"]
            jobs[job_id].update(status=status, returncode=returncode, log=output)
        except Exception as error:
          with jobs_lock:
            stopped = job_id in cancel_requests
            jobs[job_id].update(
              status="stopped" if stopped else "failed", returncode=-1, log=str(error)
            )
        finally:
          with jobs_lock:
            parser_processes.pop(job_id, None)
            cancel_requests.discard(job_id)
          if config_file:
            config_file.unlink(missing_ok=True)

    def safe_path(rel: str) -> Path:
        path = (out_root / rel).resolve()
        if not path.is_relative_to(out_root):
            abort(403)
        if not path.exists():
            abort(404)
        return path

    def read_json(path: Path):
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        if request.path.startswith("/api/"):
            return jsonify({"error": error.description}), error.code
        return error

    @app.get("/")
    def index() -> Response:
        return Response(PAGE, mimetype="text/html")

    @app.get("/api/files")
    def api_files():
        """All parsed files: directories under out_root that contain result.json."""
        items = []
        if out_root.exists():
            for directory in sorted(out_root.iterdir()):
                if directory.name.startswith(".") or directory.name == "_failures":
                    continue
                result_path = directory / "result.json"
                if directory.is_dir() and result_path.exists():
                    try:
                        result = read_json(result_path)
                        missing = [
                            page["dir"]
                            for page in result.get("pages", [])
                            if page.get("dir")
                            and not (directory / page["dir"] / "layout.json").is_file()
                        ]
                        items.append(
                            {
                                "name": directory.name,
                                **result,
                                "has_markdown": (directory / "document.md").is_file(),
                                "status": (
                                    "incomplete"
                                    if missing
                                    else result.get("status", "legacy")
                                ),
                            }
                        )
                    except (json.JSONDecodeError, OSError):
                        items.append(
                            {"name": directory.name, "error": "result.json unreadable"}
                        )
        failures = []
        for record in sorted((out_root / "_failures").glob("*.json")):
            try:
                failures.append(
                    {"record": f"_failures/{record.name}", **read_json(record)}
                )
            except (json.JSONDecodeError, OSError):
                failures.append(
                    {
                        "record": f"_failures/{record.name}",
                        "error": "Unreadable failure record",
                    }
                )
        return jsonify(
            {"output_dir": str(out_root), "files": items, "failures": failures}
        )

    @app.get("/api/local/settings")
    def api_local_settings():
        local_processing_required()
        with jobs_lock:
            active_job = next((dict(job) for job in jobs.values()
                               if job["status"] in {"running", "stopping"}), None)
        return jsonify(
            {
                "input_dir": str(initial_input),
                "config_path": str(selected_config) if selected_config else "",
                "roots": [str(root) for root in allowed_roots()],
                "output_dir": str(out_root),
                "active_job": active_job,
            }
        )

    @app.get("/api/local/browse")
    def api_local_browse():
        local_processing_required()
        mode = request.args.get("mode", "input")
        if mode not in {"input", "output", "config"}:
          abort(400, "mode must be input, output or config")
        path = local_path(request.args.get("path", str(workspace_root)))
        if not path.is_dir():
            abort(400, "Selected path is not a directory")
        entries = []
        try:
            children = sorted(
                path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
            )
            for child in children:
                if child.name.startswith(".") or child.is_symlink():
                    continue
                try:
                    is_directory = child.is_dir()
                    is_config = (
                        mode == "config"
                        and child.is_file()
                        and child.suffix.lower() == ".json"
                    )
                    if is_directory or is_config:
                        local_path(str(child))
                        entries.append(
                            {
                                "name": child.name,
                                "path": str(child),
                                "kind": "directory" if is_directory else "config",
                            }
                        )
                except OSError:
                    continue
                if len(entries) >= 500:
                    break
        except OSError as error:
            abort(400, str(error))
        parents = [
            root
            for root in allowed_roots()
            if path != root and path.is_relative_to(root)
        ]
        parent = max(parents, key=lambda root: len(str(root))) if parents else None
        return jsonify(
            {
                "path": str(path),
                "parent": str(path.parent) if parent else None,
                "entries": entries,
                "truncated": len(entries) >= 500,
            }
        )

    @app.post("/api/local/process")
    def api_local_process():
        nonlocal out_root
        local_processing_required()
        body = request.get_json(silent=True) or {}
        input_value = body.get("input_dir")
        if not isinstance(input_value, str) or not input_value.strip():
            abort(400, "Input folder is required")
        input_dir = local_path(input_value)
        if not input_dir.is_dir():
            abort(400, "Input folder does not exist")
        output_value = body.get("output_dir", str(out_root))
        if not isinstance(output_value, str) or not output_value.strip():
            abort(400, "Output folder is required")
        output_dir = local_path(output_value)
        if output_dir.exists() and not output_dir.is_dir():
          abort(400, "Output path is not a directory")
        markdown = body.get("markdown", True)
        if type(markdown) is not bool:
            abort(400, "Markdown option must be a boolean")
        skip_existing = body.get("skip_existing", False)
        if type(skip_existing) is not bool:
          abort(400, "Skip-existing option must be a boolean")
        config_value = str(body.get("config_path", "")).strip()
        chosen_config = local_path(config_value) if config_value else (
          None if "config_path" in body else selected_config)
        if chosen_config and (
            chosen_config.suffix.lower() != ".json" or not chosen_config.is_file()
        ):
            abort(400, "Config file must be an existing JSON file")
        try:
            cfg = load_config(chosen_config)
            files = sorted(
                path
                for path in input_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS | {".pdf"}
            )
            cfg["input_dir"] = str(input_dir)
            cfg["output_dir"] = str(output_dir)
            cfg, _ = preflight(cfg, files)
        except (OSError, ValueError, RuntimeError) as error:
            abort(400, str(error))
        page_counts = {}
        for path in files:
            try:
                if path.suffix.lower() == ".pdf":
                    with fitz.open(path) as document:
                        page_counts[path.name] = max(1, document.page_count)
                else:
                    page_counts[path.name] = 1
            except Exception:
                page_counts[path.name] = 1
        total_pages = max(1, sum(page_counts.values()))
        with jobs_lock:
            if any(job["status"] in {"running", "stopping"} for job in jobs.values()):
                abort(409, "A parsing job is already running")
            for finished_id in list(jobs)[:-25]:
              if jobs[finished_id]["status"] not in {"running", "stopping"}:
                    jobs.pop(finished_id)
            job_id = uuid.uuid4().hex
            jobs[job_id] = {
                "id": job_id,
                "status": "running",
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "markdown": markdown,
                "skip_existing": skip_existing,
                "config_path": (
                    str(chosen_config) if chosen_config else "built-in defaults"
                ),
                "progress": {
                  "completed_pages": 0,
                  "total_pages": total_pages,
                  "current_file": "",
                  "current_page": 0,
                },
                "log": "Starting parser...",
            }
            out_root = output_dir
            threading.Thread(
              target=run_parser,
              args=(job_id, cfg, markdown, skip_existing, page_counts),
              daemon=True,
            ).start()
        return jsonify(jobs[job_id]), 202

    @app.post("/api/local/jobs/<job_id>/stop")
    def api_local_stop(job_id: str):
        local_processing_required()
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                abort(404)
            if job["status"] not in {"running", "stopping"}:
                return jsonify(job)
            process = parser_processes.get(job_id)
            if process is not None and process.poll() is not None:
                return jsonify(job)
            cancel_requests.add(job_id)
            job["status"] = "stopping"
            if process is not None:
                try:
                    process.terminate()
                except OSError:
                    pass
            return jsonify(job), 202

    @app.get("/api/local/jobs/<job_id>")
    def api_local_job(job_id: str):
        local_processing_required()
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                abort(404)
            return jsonify(job)

    @app.get("/api/layout/<name>/<page_dir>")
    def api_layout(name: str, page_dir: str):
        path = safe_path(f"{name}/{page_dir}/layout.json")
        data = read_json(path)
        data["has_overlay"] = (path.parent / "overlay.png").exists()
        data["has_page_image"] = (path.parent / "page.png").exists()
        data["has_native_vectors"] = (path.parent / "native_vectors.json").exists()
        return jsonify(data)

    @app.get("/api/vectors/<name>/<page_dir>/<rid>")
    def api_vectors(name: str, page_dir: str, rid: str):
        return jsonify(read_json(safe_path(f"{name}/{page_dir}/vectors/{rid}.json")))

    @app.get("/api/native/<name>/<page_dir>")
    def api_native(name: str, page_dir: str):
        return jsonify(read_json(safe_path(f"{name}/{page_dir}/native_vectors.json")))

    @app.get("/files/<path:rel>")
    def files(rel: str):
      path = safe_path(rel)
      options = {"mimetype": "text/markdown; charset=utf-8"} if path.suffix.lower() == ".md" else {}
      return send_file(path, as_attachment=request.args.get("download") == "1", **options)

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
  --c-drawing:#2ecc71; --c-image:#d05ce3; --c-table:#e0c341;
  --radius:10px; --font:13px/1.5 "Segoe UI","Malgun Gothic",system-ui,sans-serif;
}
*{box-sizing:border-box; margin:0}
html,body{height:100%}
body{font:var(--font); background:var(--bg); color:var(--fg); overflow:hidden}
#app{display:grid; grid-template-columns:var(--w-side,250px) 5px minmax(0,1fr) 5px var(--w-detail,340px);
  grid-template-rows:auto minmax(0,1fr); height:100dvh}
#topbar{grid-column:1/-1; display:flex; flex-wrap:wrap; align-items:center; gap:8px 16px;
  padding:8px 12px; border-bottom:1px solid var(--border); background:var(--panel)}
#viewerTitle{font-size:14px; font-weight:600; letter-spacing:0; white-space:nowrap}
#viewerTitle span{color:var(--accent)}

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
#side{background:var(--panel); border-right:1px solid var(--border); display:flex; flex-direction:column; min-width:0; min-height:0}
#languageToggle{width:42px; height:30px; flex:none; margin-left:auto; padding:0;
  border:1px solid var(--border); background:var(--panel2)}
#side .sub{padding:0 16px 10px; color:var(--fg-dim); font-size:11px; border-bottom:1px solid var(--border);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
#processPanel{flex:1 1 740px; min-width:0; display:flex; flex-wrap:wrap; align-items:center; gap:8px}
#processPanel .path-pick{width:auto; flex:1 1 180px; min-width:120px; max-width:300px}
#processOptions{display:flex; flex-wrap:wrap; gap:10px; align-items:center; font-size:12px}
#processOptions label{display:flex; gap:5px; align-items:center; cursor:pointer}
#processLogPanel{padding:8px; display:flex; flex-direction:column; flex:none; gap:4px;
  height:clamp(72px,var(--h-log,var(--h-log-default,182px)),calc(100% - 105px)); min-height:0}
#processLogPanel[hidden]{display:none}
#splitLog{display:block; touch-action:none}
#splitLog[hidden]{display:none}
#splitLog:focus-visible{outline:2px solid var(--accent); outline-offset:-2px; background:var(--accent)}
#processPanel button:disabled{opacity:.5; cursor:default}
.path-pick{display:flex; align-items:center; gap:6px; min-width:0; width:100%; border:1px solid var(--border);
  border-radius:6px; background:var(--panel2); color:var(--fg); padding:5px 7px; cursor:pointer; text-align:left}
.path-pick:hover{border-color:var(--accent)}
.path-pick strong{font-size:11px; white-space:nowrap}
.path-pick span{font-size:10px; color:var(--fg-dim); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
#processActions{display:flex; align-items:center; gap:6px}
#processButton{background:var(--accent-dim); color:var(--fg); min-width:92px; height:30px}
#processButton[data-active="true"]{background:#773338; color:#fff}
#processSummary{font-size:11px; overflow-wrap:anywhere}
#processLogHeading{font-size:10px; color:var(--fg-dim)}
#processProgress{display:flex; align-items:center; gap:8px; min-height:14px}
#processProgress[hidden]{display:none}
#processProgressTrack{position:relative; flex:1; height:5px; overflow:hidden; border-radius:99px; background:var(--border)}
#processProgressFill{height:100%; width:0; border-radius:inherit; background:var(--accent); transition:width .2s ease}
#processProgressTrack::after{position:absolute; inset:0 auto 0 -35%; width:35%; content:""; opacity:0;
  background:linear-gradient(90deg,transparent,#ffffff70,transparent)}
#processProgressTrack[data-active="true"]::after{opacity:1; animation:progressSweep 1.3s ease-in-out infinite}
@keyframes progressSweep{to{transform:translateX(390%)}}
#processProgressLabel{min-width:92px; color:var(--fg-dim); font-size:10px; text-align:right; white-space:nowrap}
@media(prefers-reduced-motion:reduce){#processProgressFill{transition:none}#processProgressTrack[data-active="true"]::after{animation:none; left:65%}}
#processStatus{font-size:11px; color:var(--fg-dim); overflow-wrap:anywhere; flex:1; min-height:0;
  overflow:auto; white-space:pre-wrap; padding:4px 6px; border:1px solid var(--border); border-radius:4px; background:var(--panel2)}
#processStatus:empty::before{content:attr(data-empty)}
#picker{position:fixed; z-index:20; inset:0; display:flex; align-items:center; justify-content:center; padding:16px;
  background:#0009}
#picker[hidden],#processPanel[hidden]{display:none}
#pickerDialog{display:flex; flex-direction:column; width:min(520px,100%); max-height:min(640px,90vh);
  background:var(--panel); border:1px solid var(--border); border-radius:8px; box-shadow:0 16px 50px #0008}
#pickerHead{display:flex; align-items:center; gap:10px; padding:12px; border-bottom:1px solid var(--border)}
#pickerTitle{font-size:13px; font-weight:600; flex:1}
#pickerPath{padding:8px 12px; color:var(--fg-dim); font-size:11px; overflow-wrap:anywhere; border-bottom:1px solid var(--border)}
#pickerRoots{display:flex; flex-wrap:wrap; gap:5px; padding:8px 12px}
#pickerList{overflow:auto; padding:4px 8px 8px; min-height:100px}
.picker-entry{display:flex; width:100%; gap:8px; padding:7px 8px; color:var(--fg); background:transparent;
  border:0; border-radius:5px; text-align:left; cursor:pointer; font:inherit}
.picker-entry:hover{background:var(--panel2)}
#pickerFoot{padding:10px 12px; border-top:1px solid var(--border); display:flex; justify-content:flex-end; gap:6px}
#fileList{overflow-y:auto; flex:1; min-height:0; padding:8px}
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
#main{display:flex; flex-direction:column; min-width:0; min-height:0; background:var(--bg)}
#toolbar{display:flex; align-items:center; gap:10px; flex-wrap:wrap; padding:8px 14px;
  background:var(--panel); border-bottom:1px solid var(--border)}
#crumb{font-size:12px; color:var(--fg-dim); margin-right:auto; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
#crumb b{color:var(--fg); font-weight:600}
#markdownLink{text-decoration:none}
#typeChips{display:flex; gap:5px}
#markdownView{flex:1; min-height:0; display:flex; flex-direction:column; background:var(--panel)}
#markdownView[hidden],#canvasWrap[hidden],#toolbar [hidden]{display:none}
#markdownHeader{display:flex; align-items:center; gap:8px; padding:8px 14px; border-bottom:1px solid var(--border)}
#markdownName{flex:1; min-width:0; overflow-wrap:anywhere; font-size:12px; color:var(--fg-dim)}
#markdownModes{display:flex; gap:2px; padding:2px; border:1px solid var(--border); border-radius:6px; background:var(--panel2)}
#markdownModes .tbtn{padding:3px 8px}
#markdownDownload{display:flex; align-items:center; justify-content:center; width:30px; height:30px;
  border:1px solid var(--border); text-decoration:none; font-size:18px}
#markdownContent{flex:1; min-height:0; padding:18px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere;
  font:13px/1.7 Consolas,"Malgun Gothic",monospace; tab-size:4; color:var(--fg)}
#markdownRendered{flex:1; min-height:0; padding:18px; overflow:auto; overflow-wrap:anywhere; line-height:1.7}
#markdownRendered[hidden],#markdownContent[hidden]{display:none}
#markdownRendered h1,#markdownRendered h2,#markdownRendered h3,#markdownRendered h4,#markdownRendered h5,#markdownRendered h6{margin:1em 0 .55em; line-height:1.35}
#markdownRendered h1:first-child,#markdownRendered h2:first-child{margin-top:0}
#markdownRendered p{margin:0 0 1em}
#markdownRendered table{border-collapse:collapse; margin:0 0 1em; max-width:100%; display:block; overflow:auto}
#markdownRendered th,#markdownRendered td{padding:5px 9px; border:1px solid var(--border); text-align:left; white-space:pre-wrap}
#markdownRendered th{background:var(--panel2); color:var(--fg)}
#markdownRendered hr{border:0; border-top:1px solid var(--border); margin:1em 0}
.mditem{padding-left:28px}
.markdown-file{flex:none; border:1px solid var(--border); font-size:10px; padding:2px 5px}
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
#ov .rgn rect.cell{stroke-width:1; opacity:.7; pointer-events:none}
#ov .rgn:hover rect.cell, #ov .rgn.sel rect.cell{fill:transparent}
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
#selection{font:600 11px Consolas,monospace; color:var(--accent); white-space:nowrap}
#rlist{overflow-y:auto; flex:1; min-height:60px; padding:6px}
.rrow{display:flex; align-items:center; gap:8px; padding:6px 8px; border-radius:8px; cursor:pointer; font-size:12px}
.rrow:hover{background:var(--panel2)}
.rrow.sel{background:var(--accent-dim); box-shadow:inset 3px 0 var(--accent); outline:1px solid var(--accent); outline-offset:-1px}
.rrow.sel .id,.rrow.sel .snip{color:var(--fg); font-weight:600}
.rrow:focus-visible{outline:2px solid var(--accent); outline-offset:-2px}
.rrow .dot{width:9px; height:9px; border-radius:3px; flex:none}
.rrow .id{font-family:Consolas,monospace; color:var(--fg-dim); flex:none}
.rrow .snip{flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--fg-dim)}
.rrow .cf{flex:none; font-size:10.5px; color:var(--fg-dim)}
#rdetail{overflow-y:auto; flex:none; height:var(--h-detail,45%); min-height:100px; display:none; background:var(--panel)}
#rdetail.show{display:block}
#detail.has-detail #splitD{display:block}
#rdetail .inner{padding:12px 14px 16px}
#rdetail h3{font-size:13px; display:flex; align-items:center; gap:8px; margin-bottom:8px}
#rdetail h3{flex-wrap:wrap}
#rdetail td{overflow-wrap:anywhere}
#typeChips{flex-wrap:wrap}
#rdetail h3 .badge{font-size:10.5px; padding:2px 8px; border-radius:999px; color:#fff; font-weight:600}
#rdetail h3 .zoombtn{margin-left:auto}
#rdetail table{width:100%; border-collapse:collapse; font-size:11.5px; margin-bottom:10px}
#rdetail td{padding:3px 0; vertical-align:top}
#rdetail td:first-child{color:var(--fg-dim); width:88px; white-space:nowrap}
#rdetail .txtbox{background:var(--panel2); border:1px solid var(--border); border-radius:8px;
  padding:8px 10px; font-size:12.5px; margin-bottom:10px; word-break:break-all; white-space:pre-wrap}
#rdetail .cellgrid-wrap{overflow-x:auto; margin-bottom:10px}
#rdetail table.cellgrid{width:auto; min-width:100%; font-size:11.5px}
#rdetail table.cellgrid td{border:1px solid var(--border); background:var(--panel2);
  padding:4px 7px; color:var(--fg); width:auto; white-space:normal; word-break:break-all}
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
@media(max-width:1100px){
  #topbar{display:grid; grid-template-columns:minmax(0,1fr) auto}
  #processPanel{grid-column:1/-1; grid-row:2}
  #languageToggle{grid-column:2; grid-row:1}
}
@media(max-width:800px){
  body{overflow:auto}
  #app{grid-template-columns:minmax(0,1fr); grid-template-rows:auto 300px 480px 320px;
    height:auto; min-height:100dvh}
  .vsplit{display:none}
  #side,#detail{min-height:0}
  #side .sub{display:none}
  #topbar{padding:6px 8px; gap:6px 10px}
  #processPanel{gap:5px}
  #processLogPanel{--h-log-default:120px}
  .path-pick{padding:3px 6px}
  #processStatus{padding:4px 5px}
  #toolbar{gap:5px; padding:6px}
  #crumb{flex-basis:100%}
}
</style>
</head>
<body>
<div id="app">
  <header id="topbar">
    <h1 id="viewerTitle">Doc Layout Parser <span>Viewer</span></h1>
  <nav id="processPanel" aria-label="Processing" hidden>
    <button class="path-pick" id="inputPathButton" type="button"><strong id="inputPathLabel">입력</strong><span id="inputPath"></span></button>
    <button class="path-pick" id="outputPathButton" type="button"><strong id="outputPathLabel">출력</strong><span id="outputPath"></span></button>
    <button class="path-pick" id="configPathButton" type="button"><strong id="configPathLabel">설정</strong><span id="configPath"></span></button>
    <div id="processOptions">
      <label><input id="markdownOption" type="checkbox" checked><span id="markdownOptionLabel">Markdown 생성</span></label>
      <label><input id="skipExistingOption" type="checkbox"><span id="skipExistingLabel">기존 결과 건너뛰기</span></label>
    </div>
    <div id="processActions">
      <button class="tbtn" id="processButton" type="button">처리</button>
      <span id="processSummary" role="status" aria-live="polite"></span>
    </div>
  </nav>
    <button class="tbtn" id="languageToggle" type="button" aria-label="Switch to English" title="Switch to English">EN</button>
  </header>
  <aside id="side">
    <div id="processLogPanel" hidden>
      <div id="processLogHeading">처리 로그</div>
      <div id="processProgress" hidden>
        <div id="processProgressTrack" role="progressbar" aria-label="처리 진행률" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
          <div id="processProgressFill"></div>
        </div>
        <span id="processProgressLabel"></span>
      </div>
      <pre id="processStatus" data-empty="로그가 아직 없습니다." role="status" aria-live="polite"></pre>
    </div>
    <div class="hsplit" id="splitLog" role="separator" tabindex="0" aria-orientation="horizontal"
      aria-controls="processLogPanel" aria-label="처리 로그" hidden></div>
    <div class="sub" id="outdir"></div>
    <div id="fileList"></div>
  </aside>

  <div class="vsplit" id="splitL" title="드래그로 크기 조절, 더블클릭으로 초기화"></div>

  <section id="main">
    <div id="toolbar">
      <div id="crumb">파일을 선택하세요</div>
      <div class="tgroup" role="tablist" aria-label="Document view">
        <button class="tbtn on" id="documentTab" type="button" role="tab" aria-selected="true" aria-controls="canvasWrap">문서</button>
        <button class="tbtn" id="markdownLink" type="button" role="tab" aria-selected="false" aria-controls="markdownView" disabled>Markdown</button>
      </div>
      <div class="tgroup" id="baseGroup">
        <button class="tbtn on" data-base="page">원본</button>
        <button class="tbtn" data-base="overlay">오버레이</button>
      </div>
      <div class="tgroup" id="layerGroup">
        <button class="tbtn on" id="lyBbox">영역 박스</button>
        <button class="tbtn" id="lyVec">벡터</button>
        <button class="tbtn" id="lyNative" style="display:none">PDF 벡터</button>
      </div>
      <div id="typeChips"></div>
      <div class="tgroup" id="zoomGroup">
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
    <section id="markdownView" role="tabpanel" aria-labelledby="markdownLink" hidden>
      <div id="markdownHeader"><span id="markdownName">document.md</span>
        <div id="markdownModes" role="group" aria-label="Markdown 표시 방식">
          <button class="tbtn on" id="markdownRenderedMode" type="button" aria-pressed="true">렌더링</button>
          <button class="tbtn" id="markdownTextMode" type="button" aria-pressed="false">텍스트</button>
        </div>
        <a class="tbtn" id="markdownDownload" download="document.md" aria-label="Markdown 다운로드" title="Markdown 다운로드">&#8595;</a>
      </div>
      <article id="markdownRendered" tabindex="0"></article>
      <pre id="markdownContent" tabindex="0"></pre>
    </section>
  </section>

  <div class="vsplit" id="splitR" title="드래그로 크기 조절, 더블클릭으로 초기화"></div>

  <aside id="detail">
    <div id="dhead"><span id="regionHeading">레이아웃 영역</span><span id="selection" aria-live="polite"></span><span class="n" id="rcount"></span></div>
    <div id="rlist" role="listbox" aria-label="레이아웃 영역"></div>
    <div class="hsplit" id="splitD" title="드래그로 크기 조절, 더블클릭으로 초기화"></div>
    <div id="rdetail"><div class="inner" id="rdetailInner"></div></div>
  </aside>
</div>
<div id="picker" hidden>
  <section id="pickerDialog" role="dialog" aria-modal="true" aria-labelledby="pickerTitle">
    <div id="pickerHead"><span id="pickerTitle"></span><button class="tbtn" id="pickerClose" type="button" aria-label="닫기">✕</button></div>
    <div id="pickerRoots"></div>
    <div id="pickerPath"></div>
    <div id="pickerList"></div>
    <div id="pickerFoot"><button class="tbtn" id="pickerChoose" type="button"></button></div>
  </section>
</div>

<script>
"use strict";
const TYPE_COLORS = {text:"#3b9dff", dimension:"#ff5252", annotation:"#ffa726", drawing:"#2ecc71", image:"#d05ce3", table:"#e0c341"};
const TYPE_LABELS = {text:"텍스트", dimension:"치수", annotation:"주석", drawing:"도면", image:"이미지", table:"표"};
const EN = {
  "텍스트":"Text", "치수":"Dimension", "주석":"Annotation", "도면":"Drawing", "이미지":"Image", "표":"Table",
  "레이아웃 영역":"Layout Regions", "파일을 선택하세요":"Select a file", "원본":"Original", "오버레이":"Overlay",
  "Markdown 문서 열기":"Open Markdown document",
  "문서":"Document", "Markdown 다운로드":"Download Markdown", "Markdown 파일이 없습니다":"No Markdown file available",
  "렌더링":"Rendered", "텍스트":"Text", "Markdown 표시 방식":"Markdown display mode",
  "Markdown 불러오는 중…":"Loading Markdown…",
  "영역 박스":"Boxes", "벡터":"Vectors", "PDF 벡터":"PDF Vectors", "맞춤":"Fit", "화면 맞춤":"Fit to canvas",
  "드래그로 크기 조절, 더블클릭으로 초기화":"Drag to resize; double-click to reset",
  "선택 {id}":"Selected {id}", "파싱 결과가 없습니다.":"No parsed results.", "페이지 {number}":"Page {number}",
  "{count} 영역":"{count} regions", "폴리라인 {count}개":"{count} polylines", "래스터 이미지":"Raster image",
  "표 {rows}행×{cols}열":"Table {rows} × {cols}", "표 구조":"Table structure", "영역으로 확대":"Zoom to region",
  "닫기":"Close", "점수":"Score", "분류 방법":"Classification", "단어 수":"Words", "폴리라인":"Polylines",
  "{rows}행 × {cols}열, 셀 {cells}개":"{rows} rows × {cols} columns, {cells} cells",
  "SVG 열기":"Open SVG", "영역 이미지":"Region image", "벡터 로딩 중…":"Loading vectors…",
  "폴리라인 {count}개 · 연결그룹 {groups}개":"{count} polylines · {groups} connected groups",
  "벡터 로드 실패: {error}":"Vector load failed: {error}", "로드 실패: {error}":"Load failed: {error}",
  "실패":"Failed", "완료":"Complete", "불완전":"Incomplete", "이전 형식":"Legacy", "읽기 오류":"Unreadable",
  "처리 중":"Running", "소속 / 의미":"Context / meaning", "표 내부":"In table", "도면 내부":"In drawing",
  "지표":"Metrics", "성공":"OK", "생략":"Skipped",
  "입력":"Input", "설정":"Config", "처리":"Process", "처리 중":"Processing",
  "출력":"Output", "출력 폴더 선택":"Select output folder", "처리 로그":"Processing log",
  "로그가 아직 없습니다.":"No processing log yet.",
  "처리 진행률":"Processing progress", "페이지 수 계산 중":"Counting pages...",
  "페이지 {completed}/{total}":"Page {completed}/{total}",
  "Markdown 생성":"Generate Markdown", "기존 결과 건너뛰기":"Skip existing results",
  "멈춤":"Stop", "중지 중":"Stopping", "중지됨":"Stopped", "시작 중":"Starting",
  "완료":"Complete", "실패":"Failed", "입력 폴더 선택":"Select input folder",
  "설정 파일 선택":"Select config file", "현재 폴더 선택":"Use this folder",
  "설정 파일 선택 해제":"Use built-in defaults", "폴더를 선택하세요":"Choose a folder",
  "폴더가 비어 있습니다":"No folders or config files", "폴더 선택 기능은 로컬 실행에서만 사용할 수 있습니다":"Folder selection is only available in local mode",
  "지원 입력 {count}개":"{count} supported inputs", "입력 파일이 없습니다":"No supported input files",
  "설정 오류":"Configuration error", "파일을 읽을 수 없습니다":"Cannot read folder",
  "현재 폴더":"Current folder", "로컬 입력 처리":"Local input processing", "기본 설정":"Built-in defaults"
};
const STATUS_LABELS = {failed:"실패", complete:"완료", incomplete:"불완전", legacy:"이전 형식", unreadable:"읽기 오류",
  running:"처리 중", stopping:"중지 중", stopped:"중지됨", starting:"시작 중",
  ok:"성공", skipped:"생략", in_table:"표 내부", in_drawing:"도면 내부"};
let language = "ko";
try{ if(localStorage.getItem("viewerLanguage") === "en") language = "en"; }catch(error){}
function tr(text, values={}){
  return (language === "en" ? EN[text] || text : text).replace(/\{(\w+)\}/g, (_, key) => values[key] ?? "");
}
function statusLabel(value){ return tr(STATUS_LABELS[value] || value); }
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
  documentView: "page",
  markdownMode: "rendered", markdownText: "",
  files: [], file: null, page: null, layout: null, selId: null,
  base: "page", layers: {bbox:true, vec:false, native:false},
  types: {text:true, dimension:true, annotation:true, drawing:true, image:true, table:true},
  view: {x:0, y:0, k:1},
  vecCache: {}, nativeCache: {}, detailTab: "crop",
  fileData: null,
};

function updateSelectionLabel(){
  $("#selection").textContent = state.selId ? tr("선택 {id}", {id:state.selId}) : "";
}
function setLanguage(value){
  if(value !== "ko" && value !== "en") return;
  language = value;
  try{ localStorage.setItem("viewerLanguage", value); }catch(error){}
  document.documentElement.lang = value;
  const languageButton = $("#languageToggle");
  languageButton.textContent = value === "ko" ? "EN" : "한";
  languageButton.lang = value === "ko" ? "en" : "ko";
  languageButton.title = value === "ko" ? "Switch to English" : "한국어로 전환";
  languageButton.setAttribute("aria-label", languageButton.title);
  $("#markdownDownload").title = tr("Markdown 다운로드");
  $("#markdownDownload").setAttribute("aria-label", tr("Markdown 다운로드"));
  $("#markdownModes").setAttribute("aria-label", tr("Markdown 표시 방식"));
  $("#markdownRenderedMode").textContent = tr("렌더링");
  $("#markdownTextMode").textContent = tr("텍스트");
  $("#markdownLink").title = tr($("#markdownLink").disabled ? "Markdown 파일이 없습니다" : "Markdown 문서 열기");
  for(const [selector,text] of Object.entries({
    '#documentTab':"문서", '#regionHeading':"레이아웃 영역",'[data-base="page"]':"원본",'[data-base="overlay"]':"오버레이",
    '#lyBbox':"영역 박스",'#lyVec':"벡터",'#lyNative':"PDF 벡터",'#zoomFit':"맞춤",
    '#inputPathLabel':"입력",'#outputPathLabel':"출력",'#configPathLabel':"설정",'#processButton':"처리",
    '#processLogHeading':"처리 로그", '#markdownOptionLabel':"Markdown 생성", '#skipExistingLabel':"기존 결과 건너뛰기"
  })) $(selector).textContent = tr(text);
  renderProcessingControls();
  $("#processStatus").dataset.empty = tr("로그가 아직 없습니다.");
  $("#splitLog").setAttribute("aria-label", tr("처리 로그"));
  $("#zoomFit").title = tr("화면 맞춤");
  $("#rlist").setAttribute("aria-label", tr("레이아웃 영역"));
  $("#empty").textContent = tr("파일을 선택하세요");
  document.querySelectorAll(".vsplit,.hsplit").forEach(node => node.title = tr("드래그로 크기 조절, 더블클릭으로 초기화"));
  if(!state.layout) $("#crumb").textContent = tr("파일을 선택하세요");
  if(state.fileData) renderFiles();
  buildTypeChips(); renderRegions(); renderList(); updateSelectionLabel();
  if(state.selId && $("#rdetail").classList.contains("show")) selectRegion(state.selId);
}
$("#languageToggle").onclick = () => setLanguage(language === "ko" ? "en" : "ko");
function setDocumentView(view){
  state.documentView = view;
  const markdown = view === "markdown";
  $("#canvasWrap").hidden = markdown;
  $("#markdownView").hidden = !markdown;
  for(const id of ["baseGroup", "layerGroup", "typeChips", "zoomGroup"]) $("#" + id).hidden = markdown;
  $("#documentTab").classList.toggle("on", !markdown);
  $("#documentTab").setAttribute("aria-selected", String(!markdown));
  $("#markdownLink").classList.toggle("on", markdown);
  $("#markdownLink").setAttribute("aria-selected", String(markdown));
  if(!markdown && state.layout) requestAnimationFrame(fitView);
}
async function openMarkdown(file = state.file){
  const info = state.files.find(item => item.name === file);
  if(!info?.has_markdown) return;
  setDocumentView("markdown");
  $("#markdownName").textContent = `${info.source_file || file} / document.md`;
  $("#markdownDownload").href = fileUrl(`${file}/document.md`) + "?download=1";
  $("#markdownContent").textContent = tr("Markdown 불러오는 중…");
  try{
    const response = await fetch(fileUrl(`${file}/document.md`));
    if(!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = new TextDecoder("utf-8").decode(await response.arrayBuffer());
    if(state.file === file){
      state.markdownText = text;
      $("#markdownContent").textContent = text;
      renderMarkdown(text);
      setMarkdownMode(state.markdownMode);
    }
  }catch(error){
    if(state.file === file) $("#markdownContent").textContent = tr("로드 실패: {error}", {error:error.message});
  }
}
function markdownCells(line){
  let value = line.trim();
  if(value.startsWith("|")) value = value.slice(1);
  if(value.endsWith("|")) value = value.slice(0, -1);
  const cells = []; let cell = "", escaped = false;
  for(const char of value){
    if(char === "|" && !escaped){ cells.push(cell.trim()); cell = ""; }
    else if(char === "\\" && !escaped) escaped = true;
    else { cell += char; escaped = false; }
  }
  cells.push(cell.trim());
  return cells;
}
function renderMarkdown(text){
  const target = $("#markdownRendered");
  target.replaceChildren();
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const isTableRule = line => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
  for(let index = 0; index < lines.length;){
    const line = lines[index];
    if(!line.trim()){ index++; continue; }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if(heading){
      const node = document.createElement(`h${heading[1].length}`);
      node.textContent = heading[2]; target.appendChild(node); index++; continue;
    }
    if(index + 1 < lines.length && line.includes("|") && isTableRule(lines[index + 1])){
      const table = document.createElement("table"), head = table.createTHead().insertRow();
      for(const value of markdownCells(line)){ const cell = document.createElement("th"); cell.textContent = value; head.appendChild(cell); }
      const body = table.createTBody(); index += 2;
      while(index < lines.length && lines[index].trim() && lines[index].includes("|")){
        const row = body.insertRow();
        for(const value of markdownCells(lines[index])){ const cell = row.insertCell(); cell.textContent = value; }
        index++;
      }
      target.appendChild(table); continue;
    }
    if(/^\s*([-*_])\1\1+\s*$/.test(line)){ target.appendChild(document.createElement("hr")); index++; continue; }
    const paragraph = document.createElement("p"), parts = [];
    while(index < lines.length && lines[index].trim() && !/^(#{1,6})\s+/.test(lines[index])){
      if(index + 1 < lines.length && lines[index].includes("|") && isTableRule(lines[index + 1])) break;
      parts.push(lines[index].trim()); index++;
    }
    paragraph.textContent = parts.join(" ");
    if(paragraph.textContent) target.appendChild(paragraph);
  }
}
function setMarkdownMode(mode){
  state.markdownMode = mode === "text" ? "text" : "rendered";
  const rendered = state.markdownMode === "rendered";
  $("#markdownRendered").hidden = !rendered;
  $("#markdownContent").hidden = rendered;
  $("#markdownRenderedMode").classList.toggle("on", rendered);
  $("#markdownRenderedMode").setAttribute("aria-pressed", String(rendered));
  $("#markdownTextMode").classList.toggle("on", !rendered);
  $("#markdownTextMode").setAttribute("aria-pressed", String(!rendered));
}
$("#markdownRenderedMode").onclick = () => setMarkdownMode("rendered");
$("#markdownTextMode").onclick = () => setMarkdownMode("text");
$("#documentTab").onclick = () => setDocumentView("page");
$("#markdownLink").onclick = () => openMarkdown();

async function jget(url){ const r = await fetch(url); if(!r.ok) throw new Error(url+" -> "+r.status); return r.json(); }
const fileUrl = rel => "/files/" + rel.split("/").map(enc).join("/");

const processing = {settings:null, mode:"input", currentPath:"", configChoice:"", jobId:null, status:"", progress:null, timer:null};
function renderProcessingControls(){
  const active = ["starting", "running", "stopping"].includes(processing.status);
  const button = $("#processButton");
  button.textContent = tr(processing.status === "stopping" ? "중지 중" : processing.jobId ? "멈춤" : "처리");
  button.dataset.active = String(active);
  button.disabled = processing.status === "starting" || processing.status === "stopping";
  $("#processSummary").textContent = processing.status ? statusLabel(processing.status) : "";
  document.querySelectorAll("#processPanel .path-pick, #processOptions input").forEach(control => control.disabled = active);
  renderProcessingProgress();
}
function renderProcessingProgress(){
  const row = $("#processProgress");
  const progress = processing.progress;
  const active = ["starting", "running", "stopping"].includes(processing.status);
  if(!progress && !active){ row.hidden = true; return; }
  row.hidden = false;
  const completed = Math.max(0, Number(progress?.completed_pages) || 0);
  const total = Math.max(0, Number(progress?.total_pages) || 0);
  const percent = total ? Math.min(100, Math.round(completed * 100 / total)) : 0;
  const track = $("#processProgressTrack");
  track.dataset.active = String(active);
  track.setAttribute("aria-label", tr("처리 진행률"));
  track.setAttribute("aria-valuenow", String(percent));
  const state = processing.status === "complete" ? tr("완료") :
    processing.status === "failed" ? tr("실패") :
    processing.status === "stopped" ? tr("중지됨") : tr("처리 중");
  const detail = total ? tr("페이지 {completed}/{total}", {completed, total}) : tr("페이지 수 계산 중");
  const label = total ? `${state} · ${detail}` : detail;
  track.setAttribute("aria-valuetext", label);
  $("#processProgressFill").style.width = `${percent}%`;
  $("#processProgressLabel").textContent = label;
}
function showSelectedPaths(){
  $("#inputPath").textContent = processing.settings?.input_dir || "";
  $("#inputPath").title = processing.settings?.input_dir || "";
  const configPath = processing.settings?.config_path || tr("기본 설정");
  $("#configPath").textContent = configPath;
  $("#configPath").title = configPath;
  $("#outputPath").textContent = processing.settings?.output_dir || "";
  $("#outputPath").title = processing.settings?.output_dir || "";
}
async function initLocalProcessing(){
  try{
    processing.settings = await jget("/api/local/settings");
    $("#processPanel").hidden = false;
    $("#processLogPanel").hidden = false;
    $("#splitLog").hidden = false;
    updateLogSplitter();
    showSelectedPaths();
    $("#inputPathButton").onclick = () => openPicker("input");
    $("#outputPathButton").onclick = () => openPicker("output");
    $("#configPathButton").onclick = () => openPicker("config");
    $("#processButton").onclick = () => processing.jobId ? stopProcessing() : startProcessing();
    $("#pickerClose").onclick = closePicker;
    $("#picker").addEventListener("click", event => { if(event.target.id === "picker") closePicker(); });
    $("#pickerChoose").onclick = choosePickerPath;
    if(processing.settings.active_job){
      const job = processing.settings.active_job;
      Object.assign(processing.settings, {input_dir:job.input_dir, output_dir:job.output_dir,
        config_path:job.config_path});
      $("#markdownOption").checked = job.markdown;
      $("#skipExistingOption").checked = job.skip_existing;
      showSelectedPaths();
      processing.jobId = job.id;
      processing.status = job.status;
      processing.progress = job.progress || null;
      renderProcessingControls();
      await pollProcessing(job.id);
    }
  }catch(error){
    if(error.message.includes("404") || error.message.includes("403")) return;
    console.error("Local processing settings failed", error);
  }
}
function closePicker(){ $("#picker").hidden = true; }
async function openPicker(mode){
  processing.mode = mode;
  processing.configChoice = "";
  const folderLabel = mode === "input" ? "입력" : "출력";
  $("#pickerTitle").textContent = tr(mode === "config" ? "설정 파일 선택" : `${folderLabel} 폴더 선택`);
  $("#pickerChoose").textContent = tr(mode === "config" ? "기본 설정" : "현재 폴더 선택");
  $("#picker").hidden = false;
  const roots = processing.settings.roots;
  const rootBox = $("#pickerRoots"); rootBox.innerHTML = "";
  for(const root of roots){
    const button = el("button", {class:"smallbtn", type:"button"}, esc(root));
    button.onclick = () => browsePicker(root);
    rootBox.appendChild(button);
  }
  const initial = mode === "input" ? processing.settings.input_dir : mode === "output" ?
    processing.settings.output_dir :
    (processing.settings.config_path ? processing.settings.config_path.replace(/[\\/][^\\/]+$/, "") : roots[0]);
  try{ await browsePicker(initial); }
  catch(error){ $("#pickerList").textContent = `${tr("파일을 읽을 수 없습니다")}: ${error.message}`; }
}
async function browsePicker(path){
  processing.currentPath = path;
  const data = await jget(`/api/local/browse?mode=${processing.mode}&path=${enc(path)}`);
  processing.currentPath = data.path;
  $("#pickerPath").textContent = data.path;
  const list = $("#pickerList"); list.innerHTML = "";
  if(data.parent){
    const up = el("button", {class:"picker-entry", type:"button"}, `📁 ..`);
    up.onclick = () => browsePicker(data.parent);
    list.appendChild(up);
  }
  for(const entry of data.entries){
    const button = el("button", {class:"picker-entry", type:"button"});
    button.textContent = `${entry.kind === "directory" ? "📁" : "▤"} ${entry.name}`;
    button.onclick = () => entry.kind === "directory" ? browsePicker(entry.path) :
      (processing.configChoice = entry.path, $("#pickerPath").textContent = entry.path);
    list.appendChild(button);
  }
  if(!data.entries.length) list.appendChild(el("div", {class:"picker-entry"}, tr("폴더가 비어 있습니다")));
  if(data.truncated) list.appendChild(el("div", {class:"picker-entry"}, "500+"));
}
function choosePickerPath(){
  if(processing.mode === "input") processing.settings.input_dir = processing.currentPath;
  else if(processing.mode === "output") processing.settings.output_dir = processing.currentPath;
  else processing.settings.config_path = processing.configChoice;
  showSelectedPaths();
  closePicker();
}
async function startProcessing(){
  processing.status = "starting";
  processing.progress = null;
  renderProcessingControls();
  $("#processStatus").textContent = "";
  try{
    const response = await fetch("/api/local/process", {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({...processing.settings, markdown:$("#markdownOption").checked,
        skip_existing:$("#skipExistingOption").checked})});
    const job = await response.json();
    if(!response.ok) throw new Error(job.message || job.error || `HTTP ${response.status}`);
    processing.jobId = job.id;
    processing.status = job.status;
    processing.progress = job.progress || null;
    renderProcessingControls();
    await pollProcessing(job.id);
  }catch(error){
    showProcessingError(error);
  }
}
async function stopProcessing(){
  const jobId = processing.jobId;
  if(!jobId) return;
  processing.status = "stopping";
  renderProcessingControls();
  try{
    const response = await fetch(`/api/local/jobs/${enc(jobId)}/stop`, {method:"POST"});
    if(!response.ok) throw new Error(`Stop failed: HTTP ${response.status}`);
  }catch(error){
    processing.status = "running";
    $("#processStatus").textContent += `\n${error.message}`;
    renderProcessingControls();
  }
}
async function pollProcessing(jobId){
  let job;
  try{ job = await jget(`/api/local/jobs/${enc(jobId)}`); }
  catch(error){
    $("#processStatus").textContent += `\n${error.message}`;
    processing.timer = setTimeout(() => pollProcessing(jobId), 1500);
    return;
  }
  if(processing.jobId !== jobId) return;
  processing.status = job.status;
  processing.progress = job.progress || processing.progress;
  $("#processStatus").textContent = job.log || "";
  $("#processStatus").scrollTop = $("#processStatus").scrollHeight;
  if(["running", "stopping"].includes(job.status)){
    renderProcessingControls();
    processing.timer = setTimeout(() => pollProcessing(jobId), 900);
    return;
  }
  processing.jobId = null;
  renderProcessingControls();
  try{
    state.file = null; state.page = null; state.layout = null;
    $("#stage").style.display = "none";
    $("#markdownLink").disabled = true;
    setDocumentView("page");
    $("#empty").style.display = "flex";
    renderList(); closeDetail();
    await loadFiles();
    const latest = [...state.files].sort((left,right) =>
      Date.parse(right.started_at || "") - Date.parse(left.started_at || ""))[0];
    if(latest?.pages?.length){
      const firstPage = latest.pages.find(item => item.dir && item.status !== "failed");
      if(firstPage) await selectPage(latest.name, firstPage.dir);
    }
  }catch(error){ $("#processStatus").textContent += `\n${error.message}`; }
}
function showProcessingError(error){
  processing.status = "failed";
  processing.jobId = null;
  $("#processStatus").textContent = error.message;
  renderProcessingControls();
}

/* ---------------- sidebar ---------------- */
async function loadFiles(){
  const data = await jget("/api/files");
  state.files = data.files;
  state.fileData = data;
  renderFiles();
}
function renderFiles(){
  const data = state.fileData;
  const openFiles = new Set([...document.querySelectorAll(".fitem.open[data-file]")].map(item => item.dataset.file));
  $("#outdir").textContent = data.output_dir;
  $("#outdir").title = data.output_dir;
  const list = $("#fileList"); list.innerHTML = "";
  for(const failure of (data.failures || [])){
    const row = el("div", {class:"fitem", style:"padding:10px;overflow-wrap:anywhere"});
    row.appendChild(el("a", {href:fileUrl(failure.record), target:"_blank", title:failure.error || tr("실패")},
      `${tr("실패")}: ${esc(failure.source_file || failure.record)} (${esc(failure.error_type || "error")})`));
    list.appendChild(row);
  }
  if(!state.files.length){
    list.appendChild(el("div", {style:"padding:14px;color:var(--fg-dim);font-size:12px"},
      tr("파싱 결과가 없습니다.")));
    return;
  }
  for(const f of state.files){
    const item = el("div", {class:"fitem" + (openFiles.has(f.name) || state.file === f.name ? " open" : ""), "data-file":f.name});
    const head = el("div", {class:"fhead"},
      `<span class="arrow">▶</span><span class="name" title="${esc(f.name)}">${esc(f.source_file||f.name)}</span>
      <span class="cnt">${esc(statusLabel(f.error ? "unreadable" : f.status || "legacy"))} / ${f.num_pages ?? "?"}p</span>`);
    if(f.has_markdown){
      const markdownButton = el("button", {class:"tbtn markdown-file", type:"button", title:tr("Markdown 문서 열기"),
        "aria-label":`${f.source_file || f.name}: ${tr("Markdown 문서 열기")}`}, "MD");
      markdownButton.onclick = async event => {
        event.stopPropagation();
        const firstPage = f.pages?.find(page => page.dir && page.status !== "failed");
        if(firstPage){
          await selectPage(f.name, firstPage.dir);
          if(state.file !== f.name) return;
        }else{
          state.file = f.name; state.page = null; state.layout = null;
          $("#stage").style.display = "none";
          $("#empty").style.display = "flex";
          $("#crumb").textContent = f.source_file || f.name;
          $("#markdownLink").disabled = false;
          renderList(); closeDetail();
        }
        await openMarkdown(f.name);
      };
      head.appendChild(markdownButton);
    }
    const pages = el("div", {class:"pages"});
    for(const p of (f.pages || [])){
      if(!p.dir || p.status === "failed") continue;
      const total = p.num_regions ?? 0;
      const row = el("div", {class:"pitem" + (state.file === f.name && state.page === p.dir ? " sel" : ""), "data-file":f.name, "data-page":p.dir},
        `<span>${tr("페이지 {number}", {number:p.page})}</span><span class="rc">${tr("{count} 영역", {count:total})}</span>`);
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
  if(first && !state.file){
    list.querySelector(`.pitem[data-file="${CSS.escape(first.name)}"]`)?.parentElement.parentElement.classList.add("open");
    selectPage(first.name, first.pages[0].dir);
  }
}

/* ---------------- page load ---------------- */
async function selectPage(file, pageDir){
  const previousFile = state.file;
  state.file = file; state.page = pageDir; state.selId = null;
  updateSelectionLabel();
  state.vecCache = {}; state.nativeCache = {};
  document.querySelectorAll(".pitem").forEach(x =>
    x.classList.toggle("sel", x.dataset.file===file && x.dataset.page===pageDir));

  const layout = await jget(`/api/layout/${enc(file)}/${enc(pageDir)}`);
  if(state.file !== file || state.page !== pageDir) return;
  state.layout = layout;
  $("#stage").style.display = "";
  const fileInfo = state.files.find(item => item.name === file);
  const markdownLink = $("#markdownLink");
  markdownLink.disabled = !fileInfo?.has_markdown;
  markdownLink.title = tr(fileInfo?.has_markdown ? "Markdown 문서 열기" : "Markdown 파일이 없습니다");
  if(!fileInfo?.has_markdown) setDocumentView("page");
  else if(state.documentView === "markdown" && previousFile !== file) openMarkdown(file);
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
    for(const c of (r.table?.cells || [])){
      const [cx0,cy0,cx1,cy1] = c.bbox;
      g.appendChild(svgEl("rect", {class:"cell", x:cx0, y:cy0, width:cx1-cx0, height:cy1-cy0,
                                   stroke:color, "vector-effect":"non-scaling-stroke"}));
    }
    const t = svgEl("text", {x:x0+3, y:Math.max(14, y0-5), fill:color});
    t.textContent = `${r.id} ${tr(TYPE_LABELS[r.type] || r.type)}`;
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
      `<span class="dot" style="background:${TYPE_COLORS[t]}"></span>${tr(TYPE_LABELS[t])} ${counts[t]}`);
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
  if(r.type === "drawing") return tr("폴리라인 {count}개", {count:r.num_polylines ?? 0});
  if(r.type === "image") return tr("래스터 이미지");
  if(r.type === "table" && r.table) return tr("표 {rows}행×{cols}열", r.table);
  return "";
}
function renderList(){
  const list = $("#rlist"); list.innerHTML = "";
  const regions = (state.layout?.regions || []).filter(r => state.types[r.type]);
  $("#rcount").textContent = state.layout ? `${regions.length}/${state.layout.num_regions}` : "";
  for(const r of regions){
    const row = el("div", {class:"rrow" + (r.id===state.selId ? " sel":""), "data-id":r.id,
      role:"option", "aria-selected":String(r.id===state.selId), tabindex:"0"},
      `<span class="dot" style="background:${TYPE_COLORS[r.type]||"#999"}"></span>
       <span class="id">${r.id}</span>
       <span class="snip" title="${esc(snippet(r))}">${esc(snippet(r))}</span>
      <span class="cf">${r.confidence == null ? "N/A" : Math.round(r.confidence*100)+"%"}</span>`);
    row.onclick = () => selectRegion(r.id);
    row.onkeydown = event => { if(event.key === "Enter" || event.key === " "){ event.preventDefault(); selectRegion(r.id); } };
    list.appendChild(row);
  }
}

/* ---------------- selection & detail ---------------- */
function cellGridHtml(t){
  // Rebuild the parsed table as an HTML table (rowspan/colspan preserved).
  const cellAt = {}, covered = {};
  for(const c of t.cells) cellAt[c.row + "," + c.col] = c;
  let html = `<div class="cellgrid-wrap"><table class="cellgrid">`;
  for(let i = 0; i < t.rows; i++){
    html += "<tr>";
    for(let j = 0; j < t.cols; j++){
      if(covered[i + "," + j]) continue;
      const c = cellAt[i + "," + j];
      if(!c) continue;
      for(let a = 0; a < c.row_span; a++)
        for(let b = 0; b < c.col_span; b++) covered[(i + a) + "," + (j + b)] = true;
      html += `<td rowspan="${c.row_span}" colspan="${c.col_span}">${esc(c.text || "")}</td>`;
    }
    html += "</tr>";
  }
  return html + `</table></div><div class="cap">${tr("표 구조")}</div>`;
}

function selectRegion(id){
  state.selId = id;
  updateSelectionLabel();
  document.querySelectorAll("#rgnLayer .rgn").forEach(g => g.classList.toggle("sel", g.dataset.id===id));
  document.querySelectorAll("#rlist .rrow").forEach(row => {
    const selected = row.dataset.id === id;
    row.classList.toggle("sel", selected);
    row.setAttribute("aria-selected", String(selected));
  });
  renderDetail();
  requestAnimationFrame(() => {
    const row = document.querySelector(`#rlist .rrow[data-id="${CSS.escape(state.selId)}"]`);
    if(row) row.scrollIntoView({block:"nearest", inline:"nearest"});
  });
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
  let html = `<h3><span class="badge" style="background:${color}">${tr(TYPE_LABELS[r.type]||r.type)}</span>
      <span>${r.id}</span>
      <button class="smallbtn zoombtn" onclick="zoomToRegion('${r.id}')">${tr("영역으로 확대")}</button>
      <button class="smallbtn" onclick="closeDetail()" title="${tr("닫기")}" aria-label="${tr("닫기")}">✕</button></h3>
    <table>
      <tr><td>bbox</td><td>[${x0}, ${y0}] – [${x1}, ${y1}] &nbsp;(${x1-x0}×${y1-y0}px)</td></tr>
        <tr><td>${tr("점수")}</td><td>${r.confidence == null ? "N/A" : r.confidence.toFixed(3)} (${esc(r.confidence_kind || "legacy")})</td></tr>
      <tr><td>${tr("분류 방법")}</td><td>${esc(r.source || "-")}</td></tr>`;
      if(r.vlm) html += `<tr><td>VLM</td><td>${esc(r.vlm.model)}: ${esc(statusLabel(r.vlm.status))} ${esc(r.vlm.reason || "")}</td></tr>`;
      if(r.spatial_context) html += `<tr><td>${tr("소속 / 의미")}</td><td>${esc(statusLabel(r.spatial_context))} / ${esc(tr(TYPE_LABELS[r.semantic_type] || r.semantic_type))}</td></tr>`;
  if(r.metrics) html += `<tr><td>${tr("지표")}</td><td>${Object.entries(r.metrics)
      .map(([k,v])=>`${k}=${v}`).join(", ")}</td></tr>`;
  if(r.words?.length) html += `<tr><td>${tr("단어 수")}</td><td>${r.words.length}</td></tr>`;
  if(r.num_polylines) html += `<tr><td>${tr("폴리라인")}</td><td>${r.num_polylines}</td></tr>`;
  if(r.table) html += `<tr><td>${tr("표 구조")}</td><td>${tr("{rows}행 × {cols}열, 셀 {cells}개", {...r.table, cells:r.table.num_cells})}</td></tr>`;
  html += `</table>`;
  if(r.text) html += `<div class="txtbox">${esc(r.text)}</div>`;
  if(r.table) html += cellGridHtml(r.table);

  const hasCrop = !!r.image_file, hasVec = !!r.vector_file;
  if(hasCrop || hasVec){
    if(state.detailTab === "vec" && !hasVec) state.detailTab = "crop";
    if(state.detailTab === "crop" && !hasCrop) state.detailTab = "vec";
    html += `<div class="minitabs">`;
    if(hasCrop) html += `<button class="smallbtn ${state.detailTab==="crop"?"on":""}" onclick="setDetailTab('crop')">${tr("이미지")}</button>`;
    if(hasVec)  html += `<button class="smallbtn ${state.detailTab==="vec"?"on":""}" onclick="setDetailTab('vec')">${tr("벡터")}</button>`;
    if(r.svg_file) html += `<a class="smallbtn" style="text-decoration:none"
        href="${fileUrl(state.file+"/"+state.page+"/"+r.svg_file)}" target="_blank">${tr("SVG 열기")}</a>`;
    html += `</div><div id="mediaBox"></div>`;
  }
  inner.innerHTML = html;

  const box = $("#mediaBox");
  if(!box) return;
  if(state.detailTab === "crop" && hasCrop){
    box.innerHTML = `<div class="imgbox"><img src="${fileUrl(state.file+"/"+state.page+"/"+r.image_file)}"></div>
                     <div class="cap">${tr("영역 이미지")}</div>`;
  }else if(state.detailTab === "vec" && hasVec){
    box.innerHTML = `<div class="spin">${tr("벡터 로딩 중…")}</div>`;
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
        tr("폴리라인 {count}개 · 연결그룹 {groups}개", {count:v.num_polylines, groups:v.num_groups})));
      }catch(e){ box.innerHTML = `<div class="spin">${esc(tr("벡터 로드 실패: {error}", {error:e.message}))}</div>`; }
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
let pan = null, lastPanMoved = false;
wrap.addEventListener("click", event => {
  if(lastPanMoved){ event.preventDefault(); event.stopPropagation(); }
}, true);
wrap.addEventListener("pointerdown", ev => {
  if(ev.button !== 0) return;
  lastPanMoved = false;
  pan = {sx:ev.clientX, sy:ev.clientY, ox:state.view.x, oy:state.view.y, moved:false};
  wrap.classList.add("panning");
});
wrap.addEventListener("pointermove", ev => {
  if(!pan) return;
  const dx = ev.clientX - pan.sx, dy = ev.clientY - pan.sy;
  if(Math.abs(dx) + Math.abs(dy) > 3){ pan.moved = true; wrap.setPointerCapture(ev.pointerId); }
  if(!pan.moved) return;
  state.view.x = pan.ox + dx; state.view.y = pan.oy + dy; applyView();
});
wrap.addEventListener("pointerup", ev => {
  wrap.classList.remove("panning");
  lastPanMoved = !!pan?.moved;
  pan = null;
});
wrap.addEventListener("pointercancel", () => { pan = null; wrap.classList.remove("panning"); });
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
      bar.removeEventListener("pointercancel", up);
    };
    bar.addEventListener("pointermove", move);
    bar.addEventListener("pointerup", up);
    bar.addEventListener("pointercancel", up);
  });
  bar.addEventListener("dblclick", onReset);
}
function updateLogSplitter(){
  const bar = $("#splitLog");
  bar.setAttribute("aria-valuemin", "72");
  bar.setAttribute("aria-valuemax", String(Math.max(72, $("#side").clientHeight - 105)));
  bar.setAttribute("aria-valuenow", String(Math.round($("#processLogPanel").getBoundingClientRect().height)));
}
function resizeLog(height){
  const value = clamp(height, 72, Math.max(72, $("#side").clientHeight - 105));
  rootStyle.setProperty("--h-log", value + "px");
  saveSplit("log", Math.round(value));
  updateLogSplitter();
}
function resetLog(){
  rootStyle.removeProperty("--h-log"); saveSplit("log", null); updateLogSplitter();
}
initSplitter("splitLog", event => {
  resizeLog(event.clientY - $("#side").getBoundingClientRect().top);
}, resetLog, true);
$("#splitLog").addEventListener("keydown", event => {
  if(!["ArrowUp", "ArrowDown", "Home", "End", "Enter"].includes(event.key)) return;
  event.preventDefault();
  if(event.key === "Enter") return resetLog();
  const current = $("#processLogPanel").getBoundingClientRect().height;
  resizeLog(event.key === "Home" ? 72 : event.key === "End" ? $("#side").clientHeight - 105 :
    current + (event.key === "ArrowDown" ? 16 : -16));
});
window.addEventListener("resize", updateLogSplitter);
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
  if(s.log)     rootStyle.setProperty("--h-log", s.log + "px");
})();

setLanguage(language);
initLocalProcessing();
loadFiles().catch(e => { $("#fileList").innerHTML =
  `<div style="padding:14px;color:#ff5252;font-size:12px">${esc(tr("로드 실패: {error}", {error:e.message}))}</div>`; });
</script>
</body>
</html>
"""


# ------------------------------------------------------------------ main ---

def main():
    ap = argparse.ArgumentParser(description="Web viewer for doc_layout_parser parsing results")
    ap.add_argument("-o", "--output", default=None,
                    help="output folder to browse (default: output_dir from config.json)")
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open the web browser")
    processing_mode = ap.add_mutually_exclusive_group()
    processing_mode.add_argument("--enable-local-processing", action="store_true", default=None,
                   help="enable processing (default on loopback hosts)")
    processing_mode.add_argument("--read-only", action="store_true", help="disable local processing controls")
    args = ap.parse_args()

    if args.enable_local_processing and args.host not in {"127.0.0.1", "localhost", "::1"}:
      ap.error("--enable-local-processing requires a loopback --host")
    local_processing = not args.read_only and args.host in {"127.0.0.1", "localhost", "::1"}

    config_path = resolve_config_path(args.config, ROOT)
    cfg = load_config(config_path)
    work_root = config_path.resolve().parent if config_path else Path.cwd()
    input_path = Path(cfg["input_dir"])
    input_root = input_path if input_path.is_absolute() else work_root / input_path
    if args.output:
      output_path = Path(args.output)
      out_root = output_path if output_path.is_absolute() else Path.cwd() / output_path
    else:
      output_path = Path(cfg["output_dir"])
      out_root = output_path if output_path.is_absolute() else work_root / output_path

    if not out_root.exists():
        print(f"[WARN] output folder not found: {out_root} (run main.py first)")

    app = create_app(out_root, input_root=input_root, config_path=config_path,
             workspace_root=work_root, enable_local_processing=local_processing)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving {out_root} at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
