"""Add-in stages: user modules that run after the core parse and emit new formats.

A stage lives in addin/<name>/stage.py and exposes STAGE (an instance) or a
Stage subclass. It reads the artifacts earlier stages produced, writes its own
into a directory it owns, and returns them so the next stage can read them.

Nothing here runs unless config["stages"]["enabled"] names a stage.
"""
import copy
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path

ENTRY_FILE = "stage.py"


@dataclass(frozen=True)
class Artifact:
    """One file a stage produced. meta carries small values inline."""
    kind: str
    path: Path | None = None
    stage: str = ""
    meta: dict = field(default_factory=dict)

    def as_record(self, root: Path) -> dict:
        rel = None
        if self.path is not None:
            target = Path(self.path)
            rel = str(target.relative_to(root) if target.is_absolute() else target)
        return {"kind": self.kind, "stage": self.stage, "path": rel, "meta": self.meta}


class _Context:
    def __init__(self, root: Path, cfg: dict, artifacts: list, workdir: Path):
        self.root = root
        self.cfg = cfg
        self.workdir = workdir
        self._artifacts = artifacts

    def artifacts(self, kind: str | None = None, stage: str | None = None) -> list:
        return [a for a in self._artifacts
                if (kind is None or a.kind == kind) and (stage is None or a.stage == stage)]

    def artifact(self, kind: str) -> Artifact | None:
        found = self.artifacts(kind)
        return found[0] if found else None

    def read_json(self, relative) -> dict:
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.workdir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class PageContext(_Context):
    """One parsed page: layout plus every file export_page wrote beside it."""

    def __init__(self, page_dir: Path, layout: dict, cfg: dict, artifacts: list, workdir: Path):
        super().__init__(page_dir, cfg, artifacts, workdir)
        self.page_dir = page_dir
        self.layout = copy.deepcopy(layout)
        self.source_file = self.layout.get("source_file", "")
        self.output_key = self.layout.get("output_key", "")
        self.page_no = self.layout["page"]
        self.size = self.layout["size"]

    def regions(self, *types: str) -> list:
        regions = self.layout["regions"]
        return [r for r in regions if not types or r["type"] in types]

    def region(self, region_id: str) -> dict | None:
        return next((r for r in self.layout["regions"] if r["id"] == region_id), None)

    def vectors(self, region_id: str) -> list:
        region = self.region(region_id) or {}
        if not region.get("vector_file"):
            return []
        return self.read_json(region["vector_file"])["polylines"]

    def native_vectors(self) -> list:
        path = self.page_dir / "native_vectors.json"
        return json.loads(path.read_text(encoding="utf-8"))["polylines"] if path.exists() else []

    def table(self, region_id: str) -> dict | None:
        region = self.region(region_id) or {}
        return self.read_json(region["table_file"]) if region.get("table_file") else None

    def crop_path(self, region_id: str) -> Path | None:
        region = self.region(region_id) or {}
        return self.page_dir / region["image_file"] if region.get("image_file") else None

    def page_image(self) -> Path | None:
        path = self.page_dir / "page.png"
        return path if path.exists() else None

    def to_source(self, x: float, y: float) -> tuple:
        """Page pixels back to the input's own coordinate space."""
        m = self.layout.get("coordinate_transform", {}).get("pixel_to_source")
        if not m:
            return (x, y)
        a, b, c, d, e, f = m
        return (a * x + c * y + e, b * x + d * y + f)


class DocumentContext(_Context):
    """Every page of one input file, after all page stages have run."""

    def __init__(self, out_dir: Path, manifest: dict, pages: list, cfg: dict,
                 artifacts: list, workdir: Path):
        super().__init__(out_dir, cfg, artifacts, workdir)
        self.out_dir = out_dir
        self.manifest = copy.deepcopy(manifest)
        self.pages = pages
        self.source_file = manifest.get("source_file", "")
        self.output_key = manifest.get("output_key", "")

    def page_artifacts(self, kind: str | None = None) -> list:
        return [a for page in self.pages for a in page.artifacts(kind)]


class Stage:
    """Base add-in. Override only the hooks you need."""
    name = ""
    version = "0"
    requires: tuple = ()
    provides: tuple = ()

    def setup(self, cfg: dict) -> None:
        """Load models here; called once per run, not per page."""

    def on_page(self, ctx: PageContext) -> list:
        return []

    def on_document(self, ctx: DocumentContext) -> list:
        return []

    def teardown(self) -> None:
        """Release models here; always called, even after a failure."""


def load_stage(addin_root: Path, name: str) -> Stage:
    """Import addin/<name>/stage.py and return its Stage instance."""
    folder = addin_root / name
    entry = folder / ENTRY_FILE
    if not entry.is_file():
        raise FileNotFoundError(f"Add-in '{name}' has no {ENTRY_FILE} in {folder}")
    module_name = f"doc_layout_parser_addin_{name}"
    spec = importlib.util.spec_from_file_location(module_name, entry)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load add-in '{name}' from {entry}")
    module = importlib.util.module_from_spec(spec)
    # The add-in folder goes first so a stage can import its own helper modules.
    sys.path.insert(0, str(folder))
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if sys.path and sys.path[0] == str(folder):
            sys.path.pop(0)

    stage = getattr(module, "STAGE", None)
    if stage is None:
        # `from pipeline.stages import Stage` leaves the base class in the module
        # namespace, so look for a subclass the add-in itself defined.
        found = [value for value in vars(module).values()
                 if isinstance(value, type) and issubclass(value, Stage)
                 and value is not Stage and value.__module__ == module_name]
        if len(found) != 1:
            raise AttributeError(
                f"Add-in '{name}' must export STAGE; found {len(found)} Stage subclass(es) "
                f"in {ENTRY_FILE}")
        stage = found[0]()
    if not isinstance(stage, Stage):
        raise TypeError(f"Add-in '{name}' STAGE is not a Stage instance")
    # The folder name is the identity: it is what stages.enabled lists and what
    # stages.settings is keyed by. A class-level name that disagrees would make
    # the stage impossible to configure, so say so instead of guessing.
    if stage.name and stage.name != name:
        raise ValueError(f"Add-in in folder '{name}' declares name='{stage.name}'; "
                         f"the folder name is the stage identity, so they must match")
    stage.name = name
    return stage


class StageRunner:
    """Loads the enabled add-ins and runs them, keeping failures contained."""

    def __init__(self, cfg: dict, work_root: Path):
        self.settings = cfg.get("stages") or {}
        self.on_error = self.settings.get("on_error", "warn")
        self.stages: list = []
        self.errors: list = []
        self.records: list = []
        self._skip: set = set()
        self._spent: dict = {}
        names = list(self.settings.get("enabled") or [])
        if not names:
            return
        addin_path = Path(self.settings.get("addin_dir") or "addin")
        self.addin_root = addin_path if addin_path.is_absolute() else work_root / addin_path
        self._load(names)
        self._check_dependencies()

    @property
    def active(self) -> bool:
        return bool(self.stages)

    def _stage_cfg(self, name: str) -> dict:
        return dict((self.settings.get("settings") or {}).get(name) or {})

    def _load(self, names: list) -> None:
        for name in names:
            try:
                stage = load_stage(self.addin_root, name)
            except BaseException as error:
                self._record_error(name, "load", error)
                continue
            self.stages.append(stage)

    def _check_dependencies(self) -> None:
        """Refuse a chain whose inputs nobody produces, before any model loads."""
        available = set()
        for stage in self.stages:
            missing = [kind for kind in stage.requires if kind not in available]
            if missing:
                raise ValueError(
                    f"Add-in '{stage.name}' requires {missing} but no earlier enabled stage "
                    f"provides it; stages run in the order given by stages.enabled")
            available.update(stage.provides)

    def setup(self) -> None:
        for stage in list(self.stages):
            try:
                stage.setup(self._stage_cfg(stage.name))
            except BaseException as error:
                self._record_error(stage.name, "setup", error)
                self._skip.add(stage.name)

    def teardown(self) -> None:
        for stage in self.stages:
            try:
                stage.teardown()
            except BaseException as error:
                self._record_error(stage.name, "teardown", error)

    def _record_error(self, name: str, hook: str, error: BaseException, page=None) -> None:
        entry = {"stage": name, "hook": hook, "page": page,
                 "error_type": type(error).__name__, "error": str(error),
                 "traceback": traceback.format_exc(limit=4)}
        self.errors.append(entry)
        print(f"  [stage] {name}.{hook} failed: {type(error).__name__}: {error}")
        if self.on_error == "fail":
            raise error

    def _run_hook(self, stage: Stage, hook: str, ctx, page=None) -> list:
        if stage.name in self._skip:
            return []
        budget = self._stage_cfg(stage.name).get("budget_sec")
        started = time.monotonic()
        try:
            produced = getattr(stage, hook)(ctx) or []
        except BaseException as error:
            self._record_error(stage.name, hook, error, page)
            return []
        elapsed = time.monotonic() - started
        self._spent[stage.name] = self._spent.get(stage.name, 0.0) + elapsed
        out = []
        for artifact in produced:
            if not isinstance(artifact, Artifact):
                self._record_error(stage.name, hook, TypeError("Stages must return Artifact objects"), page)
                return []
            out.append(replace(artifact, stage=stage.name))
        self.records.append({"stage": stage.name, "version": stage.version, "hook": hook,
                             "page": page, "elapsed_sec": round(elapsed, 3),
                             "artifacts": [a.as_record(ctx.root) for a in out]})
        if budget and self._spent[stage.name] >= budget:
            print(f"  [stage] {stage.name}: budget {budget}s spent, skipping remaining pages")
            self._skip.add(stage.name)
        return out

    def run_page(self, page_dir: Path, layout: dict) -> PageContext:
        """Run every page hook in order; each stage sees what the previous produced."""
        produced: list = []
        ctx = None
        for stage in self.stages:
            workdir = page_dir / "stages" / stage.name
            workdir.mkdir(parents=True, exist_ok=True)
            ctx = PageContext(page_dir, layout, self._stage_cfg(stage.name), produced, workdir)
            produced.extend(self._run_hook(stage, "on_page", ctx, layout.get("page")))
            if not any(workdir.iterdir()):
                workdir.rmdir()
        return ctx or PageContext(page_dir, layout, {}, produced, page_dir)

    def run_document(self, out_dir: Path, manifest: dict, pages: list) -> list:
        produced: list = []
        for stage in self.stages:
            workdir = out_dir / "stages" / stage.name
            workdir.mkdir(parents=True, exist_ok=True)
            ctx = DocumentContext(out_dir, manifest, pages, self._stage_cfg(stage.name),
                                  produced, workdir)
            produced.extend(self._run_hook(stage, "on_document", ctx, None))
            if not any(workdir.iterdir()):
                workdir.rmdir()
        return produced

    def summary(self) -> dict:
        return {"stages": self.records, "stage_errors": self.errors}
