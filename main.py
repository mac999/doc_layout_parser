"""Entry point of the drawing/document parsing pipeline.

Splits every jpg/png/pdf in the input folder into pages, classifies each
page layout into text / annotation / dimension / image / drawing / table
regions, extracts information per region type (OCR text, crops,
vectorization, table structure with per-cell text) and saves the results
under output/<file_name>/page_NNN/.

Usage:
    python main.py                     # process everything in config input_dir
    python main.py -i input\\img1.jpg  # process a single file
    python main.py -c my_config.json   # use a different configuration file
"""
import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

from pipeline.config import load_config
from pipeline.loader import load_pages, maybe_upscale, IMAGE_EXTS
from pipeline.ocr import get_text_items
from pipeline.regions import detect_graphic_regions, classify_graphic_heuristic, reclassify_text_regions
from pipeline.table import try_parse_table, merge_split_tables, detect_page_tables
from pipeline.vlm import classify_with_vlm
from pipeline.vectorize import vectorize_region
from pipeline.export import export_page, save_json

ROOT = Path(__file__).parent


def process_page(page, cfg: dict, page_dir: Path) -> dict:
    """Process one page: classify the layout, parse each region type, save results."""
    regions = []
    rid = 0

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
        table = g.get("table") or try_parse_table(crop, bbox, text_items, cfg)
        if table is not None:
            label, conf, method = "table", table.pop("confidence"), "table_grid"
        elif cls_cfg["use_vlm"] and (not cls_cfg["ambiguous_only"]
                                     or conf < cls_cfg["heuristic_confidence_threshold"]):
            vlm_label = classify_with_vlm(crop, cfg)
            if vlm_label:
                label, conf, method = vlm_label, 0.9, f'vlm:{cls_cfg["provider"]}'
        rid += 1
        region = {
            "id": f"r{rid:03d}",
            "type": label,
            "bbox": [float(v) for v in bbox],
            "confidence": conf,
            "source": method,
            "metrics": metrics,
        }
        if table is not None:
            region["table"] = table
        regions.append(region)
        if label == "drawing":
            drawing_regions.append((region, g["label"]))

    # Geometric context overrides content rules: text-based regions inside a
    # table become plain text, inside a drawing become annotation.
    reclassify_text_regions(regions, cfg)

    # 4) Vectorize drawing regions (connected segments become polylines,
    #    coordinates are page pixels).
    vectors_by_region = {}
    for region, comp_label in drawing_regions:
        x0, y0, x1, y1 = [int(round(c)) for c in region["bbox"]]
        comp_mask = labels[y0:y1, x0:x1] == comp_label
        polylines = vectorize_region(page.image, region["bbox"], text_items, cfg, comp_mask)
        if polylines:
            vectors_by_region[region["id"]] = polylines
            region["num_polylines"] = len(polylines)

    return export_page(page_dir, page, regions, vectors_by_region, page.native_polylines, cfg)


def process_file(file_path: Path, cfg: dict, out_root: Path) -> dict:
    """Process one input file page by page and write a per-file summary.

    The file's output folder is recreated from scratch so no stale results
    from a previous run (renamed crops, removed tables/vectors) survive.
    """
    out_dir = out_root / file_path.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {file_path.name} ===")

    pages_summary = []
    for page in load_pages(file_path, cfg):
        page = maybe_upscale(page, cfg)
        t0 = time.time()
        page_dir = out_dir / f"page_{page.page_no:03d}"
        layout = process_page(page, cfg, page_dir)
        counts = {}
        for r in layout["regions"]:
            counts[r["type"]] = counts.get(r["type"], 0) + 1
        print(f"  page {page.page_no}: {layout['num_regions']} regions {counts} "
              f"({time.time() - t0:.1f}s)")
        pages_summary.append({
            "page": page.page_no,
            "dir": f"page_{page.page_no:03d}",
            "size": layout["size"],
            "region_counts": counts,
            "num_regions": layout["num_regions"],
        })

    result = {"source_file": file_path.name, "num_pages": len(pages_summary), "pages": pages_summary}
    save_json(out_dir / "result.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description="Drawing/document layout parsing and vectorization pipeline")
    ap.add_argument("-c", "--config", default=str(ROOT / "config.json"))
    ap.add_argument("-i", "--input", default=None,
                    help="single input file path (omit to process the whole input_dir)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = (ROOT / cfg["output_dir"]) if not Path(cfg["output_dir"]).is_absolute() else Path(cfg["output_dir"])

    if args.input:
        files = [Path(args.input)]
    else:
        in_dir = (ROOT / cfg["input_dir"]) if not Path(cfg["input_dir"]).is_absolute() else Path(cfg["input_dir"])
        files = sorted(p for p in in_dir.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS | {".pdf"})
    if not files:
        print("No input files found.")
        return 1

    ok, failed = 0, 0
    for f in files:
        try:
            process_file(f, cfg, out_root)
            ok += 1
        except Exception:
            failed += 1
            print(f"[FAILED] {f.name}\n{traceback.format_exc()}")
    print(f"\nDone: {ok} succeeded, {failed} failed -> {out_root}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
