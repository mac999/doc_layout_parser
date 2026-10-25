# Writing an add-in

An add-in is a stage that runs after the core parse. It reads what the parser
produced — regions, OCR text, polylines, table grids, crops — plus whatever
earlier stages produced, and writes its own output. The parser itself does not
change.

Nothing in `addin/` is imported unless `config.json` names it, so an add-in with
a missing dependency cannot break a normal run.

## Layout

```
addin/
  <stage_name>/
    stage.py            # required
    requirements.txt    # optional, the add-in's own dependencies
    ...                 # helper modules, model weights, whatever it needs
```

Dependencies are the add-in's own. Declare them in `requirements.txt` beside
`stage.py` and install them before enabling the stage:

```bash
pip install -r addin/dxf_export/requirements.txt
```

The bundled add-ins are also reachable as a package extra:
`pip install ".[addin]"`. An add-in whose dependency is missing raises in
`setup()`, which skips that stage for the whole run and records the reason in
`result.json`; the parse still completes, so check `stage_errors` when a stage
produces nothing.

The folder name is the stage identity. It is what `stages.enabled` lists, what
`stages.settings` is keyed by, and what the output directory is named. A class
that declares a different `name` is rejected at load time rather than becoming
impossible to configure.

`stage.py` must export `STAGE`, an instance:

```python
from pipeline.stages import Artifact, Stage

class DxfExport(Stage):
    ...

STAGE = DxfExport()
```

A single `Stage` subclass with no `STAGE` is also accepted, but exporting the
instance is clearer and is what the bundled add-ins do.

## Configuration

```json
"stages": {
  "addin_dir": "addin",
  "enabled": ["scale_calibrate", "dxf_export"],
  "on_error": "warn",
  "settings": {
    "scale_calibrate": {"read_title_block": true, "assume_scan_dpi": null},
    "dxf_export": {"units": "mm", "use_drawing_scale": true}
  }
}
```

| key | meaning |
|---|---|
| `addin_dir` | where the folders live; relative paths resolve against the config file |
| `enabled` | both the switch and the run order. Empty means no add-in is imported |
| `on_error` | `warn` records a failure and continues; `fail` aborts the document |
| `settings.<name>` | free-form, handed to that stage's `setup()`. The add-in defines its own keys |

Config validation rejects a duplicate in `enabled` and a `settings` block naming
a stage that is not enabled, so a typo surfaces instead of being ignored.

`budget_sec` is read by the runner from any stage's settings block: once a stage
has spent that many seconds in total, it is skipped for the remaining pages.

## The Stage class

```python
class Stage:
    name = ""                  # leave empty; the folder name is used
    version = "0"              # recorded in result.json
    requires: tuple = ()       # artifact kinds an earlier enabled stage must provide
    provides: tuple = ()       # artifact kinds this stage produces

    def setup(self, cfg: dict) -> None: ...
    def on_page(self, ctx: PageContext) -> list[Artifact]: ...
    def on_document(self, ctx: DocumentContext) -> list[Artifact]: ...
    def teardown(self) -> None: ...
```

Override only what you need; every hook has a no-op default.

**`setup(cfg)`** runs once per run, before the first page, and receives the
stage's own settings block. Load models here, not in `on_page` — a checkpoint
loaded per page is loaded once per page.

**`on_page(ctx)`** runs after the page directory is complete on disk. One sheet
to one DXF is a page hook.

**`on_document(ctx)`** runs after every page, before `result.json` is written,
while the add-ins still hold their models. A model assembled from several sheets
is a document hook. A stage may implement both: accumulate per page, emit once.

**`teardown()`** always runs, including after a failure.

`requires` and `provides` are checked against the order in `enabled` before any
`setup()` is called, so a chain missing an input fails immediately instead of
after the model loading and the GPU work.

## Hook order

```
page 1 ─► core parse ─► export_page ─► stage A ─► stage B ─┐
page 2 ─► core parse ─► export_page ─► stage A ─► stage B ─┤
                                                           ▼
                                        stage A.on_document ─► stage B.on_document
                                                           │
                                                result.json ◄┘
```

Within a page, stages run in `enabled` order and each sees what the previous one
returned. Across pages, every page hook completes before any document hook runs.

## Artifacts

An artifact is how one stage hands output to the next, and how the run records
what was produced.

```python
Artifact(kind="dxf", path=some_path, meta={"page": 8, "units": "mm"})
```

| field | use |
|---|---|
| `kind` | the contract. What `requires` / `provides` refer to, and what `ctx.artifact(kind)` looks up |
| `path` | the file, usually under `ctx.workdir`. May be `None` for a value-only artifact |
| `meta` | small values passed inline, so the next stage need not parse a file for one number |
| `stage` | set by the runner; do not fill it in |

Put bulk data in `path` and the few values the next stage actually needs in
`meta`. `scale_calibrate` writes `scale.json` and also puts `px_per_mm_paper`
in `meta`, so `dxf_export` reads one attribute instead of opening a file.

A hook must return a list of `Artifact`. Returning plain dicts is recorded as a
`TypeError` for that stage and the rest of the run continues.

## What a page stage can read

```python
ctx.page_dir          # Path to page_NNN/
ctx.layout            # deep copy of layout.json; mutating it changes nothing
ctx.source_file       # "img3.pdf"
ctx.output_key        # "img3.pdf-56dca20b0e1b"
ctx.page_no           # 1-based
ctx.size              # {"width": px, "height": px}
ctx.cfg               # this stage's settings block
ctx.workdir           # page_NNN/stages/<name>/, created before the hook runs

ctx.regions("text", "drawing")   # all regions when called with no arguments
ctx.region("r035")
ctx.vectors("r035")              # polylines from the region's vector_file
ctx.native_vectors()             # page-level PDF vectors, [] for images
ctx.table("r011")                # parsed cell grid, or None
ctx.crop_path("r035")            # Path to the region crop PNG, or None
ctx.page_image()                 # Path to page.png, or None

ctx.to_source(x, y)              # page pixels back to the input's own space
ctx.artifact("scale")            # first artifact of that kind from earlier stages
ctx.artifacts(kind=None, stage=None)
ctx.read_json("vectors/r035.json")   # relative to page_dir
ctx.write_json("out.json", payload)  # into workdir, returns the Path
```

A document stage has `ctx.out_dir`, `ctx.manifest` (a copy), `ctx.pages` (the
page contexts) and `ctx.page_artifacts(kind)`, plus the same `cfg`, `workdir`,
`artifact`, `artifacts`, `read_json` and `write_json`.

`ctx.layout` is a copy and the context is read-only by design: a stage annotates
by returning an artifact, never by editing the parser's result. That keeps
`layout.json` meaning exactly one thing.

`to_source` is provided because converting page pixels back to PDF points is the
one computation every exporter needs and the easiest to get wrong. The matrix is
already in `layout.json`; use the method rather than re-deriving it.

## Output

```
output/<key>/
  result.json
  stages/<name>/...                  # document-stage output
  page_001/
    layout.json  page.png  overlay.png  regions/  vectors/  tables/
    stages/<name>/...                # page-stage output
```

Write only inside `ctx.workdir`. It is created before the hook and removed again
if the hook leaves it empty, so a stage that produces nothing on a page leaves no
trace.

Core output is untouched, so a consumer that knows nothing about add-ins reads
the same files it always did.

Stages run inside the staging directory the parser publishes atomically. A crash
on page 7 of 10 leaves no partial add-in output in `output/`: the whole staged
directory is discarded and the previous result is restored.

## Failures

With `on_error: "warn"` an exception in any hook is caught, printed, and recorded
in `result.json`:

```json
"stage_errors": [
  {"stage": "boom", "hook": "on_page", "page": 1,
   "error_type": "RuntimeError", "error": "...", "traceback": "..."}
]
```

The document still completes. A failing `setup()` skips that stage for the whole
run rather than failing per page. With `on_error: "fail"` the exception
propagates and the document fails, leaving the previous output in place.

Every successful hook is recorded too, under `stages`, with the stage version,
elapsed time and the artifacts it produced.

## Models and memory

`setup()` loads once and `teardown()` releases, so cost scales with the run, not
with the page count. Two things worth knowing before loading a large model:

- The core OCR holds GPU memory for the whole run. Measured on the bundled
  sample: easyocr peaks at about 4.7 GB across a 10-page PDF, and that is live
  while your stage runs.
- Setting `ocr.gpu: false` moves OCR to the CPU and frees all of it, at roughly
  1.8x the OCR time. Neither bundled add-in uses the GPU at all.

Accept a `device` key in your settings block and honour `"cpu"` rather than
assuming CUDA.

## Worked example

`addin/scale_calibrate` and `addin/dxf_export` are a two-stage chain that turns
a drawing PDF into one DXF per page.

**`scale_calibrate`** provides `scale`. It resolves how many pixels make a
millimetre, which splits into two separate questions:

- *paper* millimetres come exactly from the PDF point transform or the render
  DPI, with no model involved. A raster scan carries no physical size at all, so
  it reports pixel units with confidence 0 unless `assume_scan_dpi` is given
- *drawing* millimetres need the scale denominator printed in the title block,
  which has to be read

Both halves are needed for real-world size, and each is reported separately with
a `confidence` a later stage can refuse to act on. On the bundled `img3.pdf` no
scale is printed, so it reports `drawing_scale: null` rather than inventing one.

The separator in `SCALE 1:100` also accepts `.`, `;` and `-`, because OCR
confuses them on drawing text — a real sheet was read as `SCALE 1.100`. A bare
`1:2` is deliberately not accepted: ratios appear throughout a drawing and would
be mistaken for a sheet scale. Requiring the keyword is what keeps the loose
separator safe.

| setting | default | meaning |
|---|---|---|
| `read_title_block` | `true` | look for a printed sheet scale |
| `assume_scan_dpi` | `null` | scanner DPI, so a raster input yields millimetres |
| `assume_drawing_scale` | `null` | denominator to use when none is printed |

**`dxf_export`** requires `scale` and provides `dxf`. It is deterministic:
polylines become `LWPOLYLINE`, text regions become `TEXT`, table cells become
closed rectangles plus their text, each on a layer named after the region type.
The y axis is flipped once, in one function.

Text uses `FIT` alignment between the region's own left and right edges. A height
taken from the bounding box alone overflowed the region for 52 of 54 strings on
one sheet, because the source glyphs are narrower than the default DXF font;
`FIT` keeps the height and condenses the width instead.

| setting | default | meaning |
|---|---|---|
| `units` | `"mm"` | `mm` or `m`; sets `$INSUNITS` |
| `use_drawing_scale` | `true` | emit real-world size when the sheet scale is known |
| `write_text`, `write_tables` | `true` | include text entities / table cell grids |
| `min_text_height_mm` | `1.0` | floor for text height |
| `text_height_ratio` | `0.8` | bbox height to cap height |
| `min_fit_width_ratio` | `0.1` | below this multiple of the height, left-align instead of fitting |
| `layers` | see `stage.py` | region type to `[layer name, colour]` |

The split is deliberate. The judgement that can be wrong lives in one small
stage that reports its own confidence; the writer stays mechanical and testable.
An IFC exporter would follow the same shape with a `semantics` stage in front,
isolating the VLM and LLM work from the deterministic writer.

Run it:

```bash
python main.py -i input/img3.pdf
```

```
=== img3.pdf ===
  [stage] 2 add-in(s): scale_calibrate, dxf_export
  [stage] scale_calibrate p1: 7.874 px/mm (pdf_point_transform), paper only
  [stage] dxf_export p1: page_001.dxf (340 polylines, 34 texts, paper)
  page 1: 42 regions {'text': 32, 'dimension': 2, 'drawing': 8}
  ...
  [stage] dxf_export: indexed 10 DXF file(s)
```

## Checklist

- the folder name, the `enabled` entry and the `settings` key are the same string
- settings have declared defaults and `setup()` rejects keys it does not know,
  the way the core validates `config.json`
- `stage.py` exports `STAGE`
- models load in `setup()`, not in `on_page()`
- hooks return a list of `Artifact`
- files are written only under `ctx.workdir`
- `requires` names every artifact kind the stage reads
- a value another stage needs is in `meta`, not only in a file
- anything that might be wrong carries a confidence the next stage can check
