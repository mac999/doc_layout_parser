"""Entry point of the drawing/document parsing pipeline.

Splits every jpg/png/pdf in the input folder into pages, classifies each
page layout into text / annotation / dimension / image / drawing / table
regions, extracts information per region type (OCR text, crops,
vectorization, table structure with per-cell text) and saves the results
under output/<filename.ext>-<path_hash>/page_NNN/.

Usage:
    python main.py                     # process everything in config input_dir
    python main.py -i input\\img1.jpg  # process a single file
    python main.py -c my_config.json   # use a different configuration file
"""
import argparse
import hashlib
import shutil
import sys
import tempfile
import time
import traceback
import uuid
import os
from contextlib import closing
from pathlib import Path

from pipeline.config import load_config, preflight, resolve_config_path, validate_config
from pipeline.loader import load_pages, maybe_upscale, IMAGE_EXTS
from pipeline.ocr import get_text_items
from pipeline.regions import detect_graphic_regions, classify_graphic_heuristic, reclassify_text_regions
from pipeline.table import (try_parse_table, merge_split_tables, detect_page_tables,
                            dedupe_table_regions)
from pipeline.vlm import VlmSession
from pipeline.vectorize import vectorize_region, native_vectors_for_region, native_coverage
from pipeline.export import export_page, save_json, build_manifest, output_validator

ROOT = Path(__file__).parent


def process_page(page, cfg: dict, page_dir: Path, vlm_session=None) -> dict:
    """Process one page: classify the layout, parse each region type, save results."""
    regions = []
    rid = 0
    vlm_session = vlm_session if vlm_session is not None else VlmSession()

    # 1) Text-based regions (text / dimension / annotation), each with a pixel bbox.
    text_items = get_text_items(page, cfg)
    for it in text_items:
        rid += 1
        regions.append({
            "id": f"r{rid:03d}",
            "type": it["type"],
            "bbox": it["bbox"],
            "text": it["text"],
            "confidence": it["confidence"],
            "source": it["source"],
            "words": it.get("words", []),
        })

    # 2) Page-level table detection from the raw ruling-line network (text
    #    masking can fragment a table's ink, the printed lines stay whole).
    page_tables = detect_page_tables(page.image, text_items, cfg)
    for pt in page_tables:
        rid += 1
        table = pt["table"]
        regions.append({
            "id": f"r{rid:03d}",
            "type": "table",
            "bbox": [float(v) for v in pt["bbox"]],
            "confidence": table.pop("confidence"),
            "source": "table_grid",
            "table": table,
        })

    # 3) Graphic region detection (page-table areas excluded), then
    #    table / drawing / image classification. Table check (ruling-line
    #    grid) runs first; non-tables keep the existing heuristic +
    #    optional VLM path unchanged.
    cls_cfg = cfg["classify"]
    graphic_regions, labels = detect_graphic_regions(
        page.image, text_items, cfg, exclude_bboxes=[pt["bbox"] for pt in page_tables])
    # Reunite tables whose ink was split into several components by text
    # masking (merged entries carry the parsed table already).
    graphic_regions = merge_split_tables(page.image, graphic_regions, text_items, cfg)
    drawing_regions = []
    for g in graphic_regions:
        bbox = g["bbox"]
        x0, y0, x1, y1 = [int(round(c)) for c in bbox]
        crop = page.image[y0:y1, x0:x1]
        label, conf, metrics = classify_graphic_heuristic(crop, cfg)
        method = "heuristic"
        vlm_result = None
        table = g.get("table") or try_parse_table(crop, bbox, text_items, cfg)
        if table is not None:
            label, conf, method = "table", table.pop("confidence"), "table_grid"
        elif cls_cfg["use_vlm"] and (not cls_cfg["ambiguous_only"]
                                     or conf < cls_cfg["heuristic_confidence_threshold"]):
            vlm_result = vlm_session.classify(crop, cfg)
            if vlm_result["label"]:
                label, conf, method = vlm_result["label"], None, f'vlm:{cls_cfg["provider"]}'
        rid += 1
        region = {
            "id": f"r{rid:03d}",
            "type": label,
            "bbox": [float(v) for v in bbox],
            "confidence": conf,
            "source": method,
            "metrics": metrics,
            "confidence_kind": "unscored" if conf is None else "heuristic_score",
        }
        if vlm_result is not None:
            region["vlm"] = vlm_result
        if table is not None:
            region["table"] = table
        regions.append(region)
        if label == "drawing":
            drawing_regions.append((region, g["label"]))

    # The page-level pass and the graphic pass can each report the same table,
    # so drop the weaker copy before anything downstream reads the regions.
    regions = dedupe_table_regions(regions)

    # Preserve semantic types independently of spatial context.
    reclassify_text_regions(regions, cfg)

    # 4) Vectorize drawing regions (connected segments become polylines,
    #    coordinates are page pixels).
    vectors_by_region = {}
    claimed_native = []
    for region, comp_label in drawing_regions:
        x0, y0, x1, y1 = [int(round(c)) for c in region["bbox"]]
        comp_mask = labels[y0:y1, x0:x1] == comp_label
        polylines = native_vectors_for_region(
            page.native_polylines, region["bbox"], text_items, cfg, comp_mask,
            exclude_bboxes=[item["bbox"] for item in regions if item["type"] == "table"],
            claimed=claimed_native)
        coverage = native_coverage(page.image, region["bbox"], text_items, cfg, comp_mask, polylines) if polylines else 0
        region["native_coverage"] = round(coverage, 3)
        if polylines and coverage >= cfg["vectorize"]["min_native_coverage"]:
            claimed_native.extend(polylines)
            region["vector_source"] = "pdf_native"
        else:
            polylines = vectorize_region(page.image, region["bbox"], text_items, cfg, comp_mask)
            region["vector_source"] = "raster"
            for polyline in polylines:
                polyline["source"] = "raster"
        if polylines:
            vectors_by_region[region["id"]] = polylines
            region["num_polylines"] = len(polylines)

    return export_page(page_dir, page, regions, vectors_by_region, page.native_polylines, cfg)


def output_key(file_path: Path) -> str:
    identity = os.path.normcase(str(file_path.resolve())).encode("utf-8")
    return f"{file_path.name}-{hashlib.sha256(identity).hexdigest()[:12]}"


def process_file(file_path: Path, cfg: dict, out_root: Path, warnings=()) -> dict:
    """Publish completed results, rolling back a failed directory replacement."""
    validate_config(cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    key = output_key(file_path)
    out_dir = out_root / key
    stage = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=out_root))
    backup = out_root / f".{key}.backup-{uuid.uuid4().hex}"
    manifest = {"schema_version": "1.0", "run_id": uuid.uuid4().hex, "status": "running",
                "source_file": file_path.name, "pages": [], "warnings": list(warnings)}
    try:
        manifest = build_manifest(file_path, cfg, manifest["run_id"], list(warnings))
        result = _process_file(file_path, cfg, stage, manifest)
        if out_dir.exists():
            out_dir.rename(backup)
        try:
            stage.rename(out_dir)
        except BaseException:
            if backup.exists():
                backup.rename(out_dir)
            raise
    except BaseException as error:
        failure_dir = out_root / "_failures"
        failure_dir.mkdir(exist_ok=True)
        save_json(failure_dir / f"{key}-{manifest['run_id']}.json", {
            **manifest, "status": "failed", "output_key": key,
            "error": str(error), "error_type": type(error).__name__,
            "backup_dir": str(backup) if backup.exists() else None,
        })
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    return result


def _process_file(file_path: Path, cfg: dict, out_dir: Path, manifest: dict) -> dict:
    print(f"\n=== {file_path.name} ===")

    pages_summary = manifest["pages"]
    vlm_session = VlmSession()
    with closing(iter(load_pages(file_path, cfg))) as pages:
        for page in pages:
            started = time.monotonic()
            summary = {"page": page.page_no, "status": "failed"}
            pages_summary.append(summary)
            try:
                page = maybe_upscale(page, cfg)
                page_dir = out_dir / f"page_{page.page_no:03d}"
                layout = process_page(page, cfg, page_dir, vlm_session)
                counts = {}
                for region in layout["regions"]:
                    counts[region["type"]] = counts.get(region["type"], 0) + 1
                summary.update(status="complete", dir=page_dir.name, size=layout["size"],
                               region_counts=counts, num_regions=layout["num_regions"],
                               warnings=layout.get("warnings", []),
                               coordinate_transform=layout.get("coordinate_transform", {}))
                manifest["num_pages"] += 1
                print(f"  page {page.page_no}: {layout['num_regions']} regions {counts}")
            except Exception as error:
                summary.update(error_type=type(error).__name__, error=str(error))
                raise
            finally:
                summary["elapsed_sec"] = round(time.monotonic() - started, 3)

    if not pages_summary:
        raise ValueError("Input contains no pages")
    manifest.update(status="complete", vlm_calls=vlm_session.calls,
                    vlm_elapsed_sec=round(vlm_session.elapsed_sec, 3))
    output_validator().validate(manifest)
    save_json(out_dir / "result.json", manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Drawing/document layout parsing and vectorization pipeline")
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("-i", "--input", default=None,
                    help="single input file path (omit to process the whole input_dir)")
    ap.add_argument("--check", action="store_true", help="check prerequisites without processing")
    args = ap.parse_args()

    config_path = resolve_config_path(args.config, ROOT)
    cfg = load_config(config_path)
    work_root = config_path.resolve().parent if config_path else Path.cwd()
    out_path = Path(cfg["output_dir"])
    out_root = out_path if out_path.is_absolute() else work_root / out_path

    if args.input:
        files = [Path(args.input)]
    else:
        in_path = Path(cfg["input_dir"])
        in_dir = in_path if in_path.is_absolute() else work_root / in_path
        files = sorted(p for p in in_dir.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS | {".pdf"})
    cfg, warnings = preflight(cfg, files)
    for warning in warnings:
        print(f"[CHECK] {warning}")
    if args.check:
        print(f"Ready: {len(files)} inputs; OCR device: {'CUDA' if cfg['ocr']['gpu'] else 'CPU'}")
        return 0

    ok, failed = 0, 0
    for f in files:
        try:
            process_file(f, cfg, out_root, warnings)
            ok += 1
        except Exception:
            failed += 1
            print(f"[FAILED] {f.name}\n{traceback.format_exc()}")
    print(f"\nDone: {ok} succeeded, {failed} failed -> {out_root}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
