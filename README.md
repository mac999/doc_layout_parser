# Doc layout parser

A parser pipeline that reads drawing/document files (jpg, png, pdf), splits them into pages, classifies the layout of each page into **text / dimension /
annotation / image / drawing / table** regions, extracts information per region type (table structure with per-cell text included), and vectorizes drawing regions into polylines. Every extracted item carries its pixel coordinates.

<p align="center">
<img src="./doc/img4.png" width="500"></img> </br>
<img src="./doc/img1.png" width="500"></img> </br>
<img src="./doc/img2.png" width="500"></img> </br>
<img src="./doc/img3.png" width="500"></img> 
</p>

## Overview

```
input/*.{jpg,png,pdf}
  +- Page split (PyMuPDF: PDF pages are rendered to images; native text
  |               and native vector paths are extracted as well)
  +- Preprocess: automatic upscaling of low-resolution pages
  +- Per-page layout analysis
  |    +- Text detection: PDF native words plus supplementary EasyOCR
  |    |    -> words are merged into lines
  |    |    -> rule-based classification: text / dimension / annotation
  |    +- Page-level table detection: long horizontal/vertical ruling
  |    |    lines are extracted from the raw page; each connected line
  |    |    network becomes a table candidate parsed by the table gates
  |    |    (tables are found whole even when text masking would cut
  |    |     their ink into fragments; detected areas are excluded from
  |    |     graphic region detection below)
  |    +- Graphic region detection: binarization + text masking +
  |    |    morphological dilation + connected components
  |    |    (ink hugging the page edge is erased first so scan/frame
  |    |     borders do not become regions)
  |    |    -> split-table merge fallback: aligned neighbouring regions
  |    |       whose gap is crossed by printed ruling lines are unioned
  |    |       when the union parses as a table
  |    |    -> table check first: ruling-line lattice detection
  |    |       (long horizontal/vertical lines, crossing coverage;
  |    |        drawings are rejected by stray-ink, boundary-span and
  |    |        cell-text-occupancy gates)
  |    |       -> row/col boundaries, merged-cell spans, per-cell text
  |    |    -> non-tables: drawing / image classification: heuristics
  |    |       (saturation, mid-tone ratio, ink ratio), optionally refined
  |    |       by a VLM (Ollama llava / OpenAI / Gemini)
    |    +- Semantic types and spatial context are stored separately
    |         (table text retains its original semantic_type)
  +- Vectorization of drawing regions:
      native paths -> clip, deduplicate, group -> coverage check
      -> raster fallback when native coverage is insufficient:
       adaptive binarization -> skeletonize -> pixel-graph tracing
       -> connected segments merged into polylines (closed loops supported)
       -> Douglas-Peucker simplification
       -> polylines sharing endpoints get the same connectivity group id
```

All coordinates in the output are **page pixel coordinates**. When a page is
upscaled, `layout.json` records `scale`, `original_size`, and the forward/inverse
affine matrices in `coordinate_transform`. PDF source coordinates are PyMuPDF's
unrotated CropBox-relative points, not raw PDF user-space coordinates.

## Output structure

Outputs use the full filename plus a normalized absolute-path hash. Processing
runs in a sibling temporary directory; completed results replace the old folder
with backup-and-rollback protection. Failure reports go to `output/_failures/`.
Legacy stem-only output folders are preserved and remain readable by the viewer.
Use one writer per input. Windows directory replacement is not a single atomic
operation: a forced termination during publication can leave a `.backup-*`
folder. Restore it manually if the published folder is absent.

```
output/<filename.ext>-<path_hash>/
  result.json                 # input hash, config, versions, page status
  page_001/
    page.png                  # rendered page image
    layout.json               # regions: id, type, bbox [x0,y0,x1,y1], text,
                              # confidence, words, source, ...
    overlay.png               # visualization (blue=text, red=dimension,
                              # orange=annotation, green=drawing, purple=image,
                              # yellow=table incl. cell boxes)
    regions/rNNN_drawing.png  # crops of drawing/image/table regions
    vectors/rNNN.json         # polylines: points [[x,y],...], closed,
                              # length_px, group
    vectors/rNNN.svg          # vectorization result as SVG
    tables/rNNN.json          # table structure: rows, cols, row/col
                              # boundaries, cells [{row, col, row_span,
                              # col_span, bbox, text}]
    native_vectors.json       # native vector paths (vector PDFs only)
```

## Installation

Prerequisites:

- Windows / Linux, Python 3.10+
- NVIDIA GPU recommended (8 GB VRAM is enough); CPU also works
- [Ollama](https://ollama.com) with a vision model — **used by the default
  `config.json`** (`classify.provider: "ollama"`): `ollama pull llava`

```powershell
# Create an environment (conda example)
conda create -n venv_lmm python=3.11
conda activate venv_lmm

# Install dependencies
pip install -r requirements.txt
```

Notes:

- EasyOCR downloads its detection/recognition models (about 120 MB) on first run.
- PyTorch with CUDA is required for GPU OCR; see https://pytorch.org for the
  wheel matching your CUDA version.
- Commercial VLM providers are optional. Set the API key via environment
  variables: `OPENAI_API_KEY` or `GOOGLE_API_KEY`.

## Usage

Before running (with the default `config.json`):

1. **Ollama must be running** with the `llava` model pulled
   (`ollama pull llava`, then make sure the Ollama service/tray app is up).
   The VLM is only called for regions where the heuristic drawing/image
   classification is uncertain (`classify.ambiguous_only: true`). If Ollama is
   not reachable, the pipeline **does not fail** — it logs a warning per call
   and keeps the heuristic result. To skip the VLM entirely, set
   `classify.use_vlm: false` in `config.json`.
2. No other external service is required. ComfyUI is **not** used by this
   project. EasyOCR runs in-process (first run downloads its models, ~120 MB).
3. Only when switching `classify.provider` to `openai` / `gemini`: set the
   `OPENAI_API_KEY` / `GOOGLE_API_KEY` environment variable.

```powershell
# Process every supported file in the input folder (config.json: input_dir)
python main.py

# Process a single file
python main.py -i input\img1.jpg

# Use a different configuration file
python main.py -c my_config.json
```

On this machine the pipeline runs under the conda environment `venv_lmm`:

```powershell
C:\Users\ktw\.conda\envs\venv_lmm\python.exe main.py
```

## Result viewer (viewer.py)

A Flask-based web app to inspect parsing results interactively — check per
file / page / region whether the parsing is correct.

```powershell
python viewer.py              # serves the output dir from config.json, opens the browser
python viewer.py -o output    # explicit output folder
python viewer.py --port 8000 --no-browser
```

Features:

- **Language**: Korean / English toggle with browser-local persistence; keeps the current page, selection, zoom and filters
- **File navigation**: sidebar lists every parsed file under `output/` with its pages
- **Page canvas**: page image with colored region bboxes (same colors as
  `overlay.png`), mouse wheel zoom / drag pan, original ⇄ overlay image toggle
- **Layers**: region boxes, vectorized polylines, native PDF vectors (vector PDFs)
- **Region list & detail**: click a region on the canvas or in the list to see
  its type, confidence, bbox, OCR text, crop image and the vectorized polylines
  rendered as SVG (colored per connectivity group), plus "zoom to region";
  table regions additionally show the parsed cell grid (merged cells preserved).
  Canvas selection highlights and scrolls to the matching list item; the selected ID appears above the list
- **Type filter**: show/hide text / dimension / annotation / drawing / image / table regions

No external service is needed for the viewer (Ollama is not used here).

![Viewer demo: language switching, linked selection, table details and vectors](doc/viewer-demo.gif)

To regenerate the demo, run the viewer on port 8003 with the sample outputs
(`img1` and `img3`), then use the optional recording dependencies:

```powershell
python -m pip install playwright Pillow
python -m playwright install chromium
python tools/record_demo.py --url http://127.0.0.1:8003
```

The recorder also checks language persistence, selection, panning, vector assets
and desktop/mobile layout. Use `--check-only` to skip writing the GIF.

## Configuration (config.json)

Every tuned threshold and heuristic weight in the pipeline lives in
`config.json` (defaults in `pipeline/config.py`); nothing input-specific is
hardcoded in the modules, so the pipeline can be adapted to other document
styles by editing the configuration only.

| Key | Description |
|---|---|
| `input_dir`, `output_dir` | Input/output folders (relative to the project root) |
| `preprocess.upscale_target_min_side` | Upscale pages whose shorter side is below this value (px). 0 disables upscaling. |
| `preprocess.max_scale` | Maximum upscale factor |
| `pdf.render_dpi` | PDF rendering resolution (default 200) |
| `pdf.use_native_text` | Include usable embedded PDF words |
| `pdf.supplement_native_with_ocr` | Default true: supplement native words with OCR. False skips OCR when usable native words exist; use only for known digital PDFs. `min_native_words` is retained for configuration compatibility. |
| `pdf.use_native_vectors` | Extract embedded PDF vector paths |
| `ocr.languages`, `ocr.gpu` | EasyOCR languages and GPU switch |
| `ocr.cpu_fallback` | Default true: use CPU if CUDA is unavailable; false fails preflight |
| `ocr.rotation_info` | Optional EasyOCR crop rotations, e.g. `[90,180,270]`; no page deskew |
| `ocr.tile_size`, `ocr.tile_overlap` | Optional tiled OCR in page pixels; defaults 0 (disabled) and 128. More tiles cost more inference time. |
| `ocr.min_confidence` | Drop OCR results below this confidence |
| `ocr.line_gap_factor`, `ocr.line_row_factor` | Word-to-line merging: max horizontal gap / vertical center distance as a multiple of the character height |
| `ocr.short_line_max_tokens`, `ocr.dim_token_ratio` | Line classification: token count treated as a "short line", and the dimension-token fraction above which a long line counts as a dimension |
| `layout.min_region_area` | Minimum graphic region size (px^2) |
| `layout.dilate_kernel` | Dilation kernel size used to merge nearby ink into regions |
| `layout.page_border_margin_px` | Erase ink within this margin of the page edges (drops scan/frame border artifacts) |
| `layout.text_in_table_type` | Table display type (default `text`); `semantic_type` retains content classification. Drawing types are preserved; `text_in_drawing_type` is a legacy setting. |
| `layout.text_region_overlap_ratio` | Fraction of a text region's area that must lie inside the graphic bbox to trigger the reclassification |
| `classify.use_vlm` | Enable VLM-based drawing/image re-classification |
| `classify.ambiguous_only` | Call the VLM only when the heuristic is uncertain |
| `classify.provider` | `ollama` / `openai` / `gemini` |
| `classify.vlm_max_image_side` | Downscale region crops to this size before sending them to the VLM |
| `classify.max_calls_per_document`, `classify.budget_sec`, `classify.max_failures` | Defaults: 20 calls, 120 seconds of accumulated call time, circuit opens after 2 consecutive failures |
| `classify.timeout_sec` | Per-request SDK timeout, capped by remaining budget; SDK retries are disabled. Transport timeouts are not a hard process deadline. |
| `classify.heuristic.*` | All thresholds/weights of the drawing/image heuristic (gray-level bounds `dark_gray_max`/`light_gray_min`, normalizers `sat_photo_norm`/`mid_photo_norm`, weights `sat_weight`/`mid_weight`/`dark_ratio_bonus`, decision point `photo_score_threshold`) |
| `table.enable` | Enable ruling-line table detection/parsing |
| `table.min_rows`, `table.min_cols` | Minimum grid size to accept a table |
| `table.min_line_length_ratio` | Ruling lines must be longer than this ratio of the region side |
| `table.min_intersection_ratio` | Required fraction of row×col boundary crossings with ink |
| `table.separator_coverage` | Ruling coverage needed on a cell border; below it cells merge into spans |
| `table.max_stray_ink_ratio` | Max non-ruling, non-text ink inside the grid; above it the region is a drawing, not a table |
| `table.min_line_kernel_px`, `table.stray_dilate_px` | Lower bound of the ruling-line morphology kernel / line-mask dilation used by the stray-ink check |
| `table.merge_max_gap_px`, `table.merge_axis_overlap` | Split-table merge: max gap between aligned neighbouring regions / required bbox alignment along the other axis |
| `table.merge_bridge_coverage` | Fraction of the gap a printed ruling line must cross for two regions to count as one split table |
| `table.page_level_detection`, `table.network_gap_px` | Detect tables from the page's connected ruling-line networks / bridge line breaks up to this size when connecting them |
| `table.min_boundary_span_ratio` | Every ruling line must cover this fraction of the grid extent (crossing lines in drawings leave short boundaries) |
| `table.min_cell_text_ratio` | Fraction of cells that must contain text; mostly-empty lattices (drawing line networks) are rejected. Lower it for form-style tables with many blank cells |
| `vectorize.simplify_epsilon` | Polyline simplification strength (px) |
| `vectorize.min_polyline_length_px` | Drop polylines shorter than this (noise filter) |
| `vectorize.native_curve_tolerance_px` | Native cubic chord-error target, default 0.25px after upscaling |
| `vectorize.min_native_coverage` | Minimum stroke-ink coverage for native-only output, default 0.9; otherwise raster fallback |

## Validation and operation

```powershell
python main.py --check
python -m unittest discover -s tests -v
python -m pipeline.evaluate reference/layout.json prediction/layout.json --min-region-f1 0.9 --max-cer 0.1
```

The evaluation thresholds above are examples, not certified document accuracy.
The core suite needs no OCR weights, GPU, VLM service, or API keys. CI runs this
suite on Python 3.11. Model inference is a separate integration check.

`pipeline.evaluate` matches regions one-to-one by type and IoU. It reports
per-type precision/recall/F1, matched IoU, Unicode character error rate (including
missing/extra regions), exact cell span-and-text scores, and matched vector
Hausdorff distance/group/closed-path counts. Optional `--min-cell-f1` and
`--max-vector-error` gates return a nonzero exit code on failure. Compare layouts
at the same pixel size; do not treat predictions as ground truth.

The output contract is [doc/result.schema.json](doc/result.schema.json).
Export also checks bbox bounds, unique IDs, cell occupancy, vector coordinates,
and referenced files. The manifest stores input SHA-256, effective configuration,
package versions, run ID, warnings and per-page status/timing. Failure records
may list completed pages from a discarded temporary run; those are diagnostic
records, not published page links. Older successful results remain untouched.

Confidence values are source-specific scores, not calibrated probabilities:
native text uses 1.0, OCR uses model scores, tables use grid coverage, and
heuristics use a separation score. VLM classifications use null (`N/A` in the
viewer), with model, status and fallback reason recorded separately. OpenAI and
Gemini receive document crops when enabled. Keep `ambiguous_only=true` and use
the local heuristic or Ollama when external transmission is inappropriate.

## Supported scope

- Tested geometry: PDF rotations 0/90/180/270, offset CropBox, upscaling and
  inverse transforms; axis-aligned ruled cells and conservative uncertain merges.
- Word boxes, not merged line hulls, are masked. Lines inside a word box or its
  padding can still be erased; there is no speculative line reconstruction.
- Bare numbers are dimension candidates, not proven measurements. `#3` is an
  annotation; radius, diameter and tolerance patterns retain their semantics in
  drawings. Table numbers display as text while preserving `semantic_type`.
- Optional rotation recognition and tiled OCR retain page coordinates. Full-page
  orientation correction, deskew and vertical reading-order reconstruction are
  not implemented. Slanted tables, borderless tables, empty forms and single-row
  or single-column tables are outside the default supported set.
- Filled-only PDF paths are excluded from native stroke output. OCR/native word
  masks remove known character outlines; undetected outlined text can remain.
  A mixed raster/native region falls back as a whole if coverage is insufficient.
- Real-document approval remains pending: independently label representative
  Korean/English/numeric text, regions, merged cells and vectors; separate tuning
  and evaluation documents. Include rotated scans, vertical dimensions, skewed
  tables and tiny text as challenge cases, with unsupported cases labeled as such.
  Set production thresholds only after measuring this held-out set.

## Technology choices

- **PyMuPDF**: the most reliable library that covers PDF page rendering plus
  native text and vector path extraction. For vector PDFs the original paths
  are exported directly, which is far more precise than raster vectorization.
- **EasyOCR**: Korean + English out of the box, GPU accelerated, light enough
  for 8 GB VRAM.
- **Hybrid layout analysis**: generic document layout models (LayoutParser,
  DocLayout-YOLO, ...) are not trained on CAD sheets and misclassify
  dimensions/annotations/figures. Content rules on OCR results combined with
  classic CV region detection and an optional VLM check is more dependable.
- **skeletonize + graph tracing** (scikit-image / OpenCV): the standard
  approach for centerline vectorization of line drawings. Outline tracers such
  as potrace produce contours, not centerlines, so they do not fit this task.
- **Ollama (llava)**: local VLM that fits in 8 GB VRAM for drawing/image
  disambiguation. If its quality is not sufficient, switch
  `classify.provider` to `openai` or `gemini`.

## Known limitations

- Very low resolution inputs (text height under ~5 px) cannot be recovered by
  OCR even after upscaling; use source scans of 3000 px or larger for reliable
  text extraction.
- The local llava model is a weak classifier; keep the heuristic as the primary
  signal or switch to a commercial VLM for higher accuracy.

# Author 
laputa99999@gmail.com