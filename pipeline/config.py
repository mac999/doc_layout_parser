"""Load config.json and merge it over the built-in defaults."""
import json
import copy
import math
import importlib.util
import os
from pathlib import Path

DEFAULTS = {
    "input_dir": "input",
    "output_dir": "output",
    "preprocess": {"upscale_target_min_side": 1400, "max_scale": 3.0},
    "pdf": {"render_dpi": 200, "use_native_text": True, "use_native_vectors": True, "min_native_words": 5,
            "supplement_native_with_ocr": True},
    "ocr": {"languages": ["ko", "en"], "gpu": True, "min_confidence": 0.30, "paragraph": False,
            "mag_ratio": 1.0, "canvas_size": 3200, "min_size": 8,
            "text_threshold": 0.6, "low_text": 0.35,
            "line_gap_factor": 1.5, "line_row_factor": 0.6,
            "short_line_max_tokens": 3, "dim_token_ratio": 0.5,
            "rotation_info": [], "tile_size": 0, "tile_overlap": 128, "cpu_fallback": True},
    "layout": {"min_region_area": 3000, "text_mask_padding": 4, "dilate_kernel": 21, "max_page_cover_ratio": 1.0,
               "page_border_margin_px": 8,
               "text_in_drawing_type": "annotation", "text_in_table_type": "text",
               "text_region_overlap_ratio": 0.5},
    "classify": {
        "use_vlm": False, "ambiguous_only": True, "heuristic_confidence_threshold": 0.75,
        "provider": "ollama", "ollama_model": "llava:latest",
        "openai_model": "gpt-4o-mini", "gemini_model": "gemini-2.0-flash", "timeout_sec": 60,
        "vlm_max_image_side": 768,
        "max_calls_per_document": 20, "max_failures": 2, "budget_sec": 120,
        "heuristic": {
            "dark_gray_max": 60, "light_gray_min": 200,
            "sat_photo_norm": 60.0, "mid_photo_norm": 0.5,
            "sat_weight": 0.45, "mid_weight": 0.45,
            "dark_ratio_bonus_threshold": 0.5, "dark_ratio_bonus": 0.10,
            "photo_score_threshold": 0.5,
        },
    },
    "vectorize": {
        "binarize_block_size": 35, "binarize_C": 11,
        "min_polyline_length_px": 8.0, "simplify_epsilon": 2.0, "bezier_samples": 8,
        "native_curve_tolerance_px": 0.25,
        "min_native_coverage": 0.9,
    },
    "table": {
        "enable": True, "min_rows": 2, "min_cols": 2,
        "line_kernel_divisor": 20, "min_line_length_ratio": 0.35,
        "boundary_merge_tol_px": 10, "min_intersection_ratio": 0.55,
        "separator_coverage": 0.45, "min_cell_size_px": 12,
        "min_grid_cover_ratio": 0.5, "max_stray_ink_ratio": 0.008,
        "min_line_kernel_px": 10, "stray_dilate_px": 5,
        "merge_max_gap_px": 200, "merge_axis_overlap": 0.8, "merge_bridge_coverage": 0.9,
        "page_level_detection": True, "network_gap_px": 3,
        "min_boundary_span_ratio": 0.6, "min_cell_text_ratio": 0.45,
    },
    "export": {"save_page_image": True, "save_region_crops": True, "save_overlay": True, "save_svg": True},
}


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base and return a new dict."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            out[k] = _merge(base[k], v)
        else:
            out[k] = v
    return out


def validate_config(cfg: dict) -> None:
    """Validate keys, types and operational ranges."""
    ratios = {"min_confidence", "text_threshold", "low_text", "dim_token_ratio",
              "max_page_cover_ratio", "text_region_overlap_ratio", "heuristic_confidence_threshold",
              "mid_photo_norm", "sat_weight", "mid_weight", "dark_ratio_bonus_threshold",
              "dark_ratio_bonus", "photo_score_threshold", "min_line_length_ratio",
              "min_intersection_ratio", "separator_coverage", "min_grid_cover_ratio",
              "max_stray_ink_ratio", "merge_axis_overlap", "merge_bridge_coverage",
              "min_boundary_span_ratio", "min_cell_text_ratio", "min_native_coverage"}
    zero_allowed = ratios | {"upscale_target_min_side", "tile_size", "tile_overlap", "text_mask_padding",
                             "page_border_margin_px", "min_native_words", "simplify_epsilon", "binarize_C",
                             "boundary_merge_tol_px", "merge_max_gap_px", "network_gap_px",
                             "max_calls_per_document", "dark_gray_max"}

    def visit(values, defaults, prefix=""):
        if not isinstance(values, dict):
            raise ValueError(f"{prefix or 'config'} must be an object")
        for key in values.keys() - defaults.keys():
            raise ValueError(f"Unknown setting: {prefix}{key}")
        for key, default in defaults.items():
            name = f"{prefix}{key}"
            if key not in values:
                raise ValueError(f"Missing setting: {name}")
            value = values[key]
            if isinstance(default, dict):
                visit(value, default, name + ".")
                continue
            expected = (int, float) if isinstance(default, float) else type(default)
            if not isinstance(value, expected) or (isinstance(value, bool) and not isinstance(default, bool)):
                raise ValueError(f"Invalid type: {name}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or value < 0 or (value == 0 and key not in zero_allowed):
                    raise ValueError(f"Invalid range: {name}")
                if key in ratios and value > 1:
                    raise ValueError(f"{name} must be between 0 and 1")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{name} must not be empty")

    visit(cfg, DEFAULTS)
    if cfg["classify"]["provider"] not in {"ollama", "openai", "gemini"}:
        raise ValueError("Invalid classify.provider")
    if cfg["ocr"]["paragraph"]:
        raise ValueError("ocr.paragraph=true is unsupported")
    if not cfg["ocr"]["languages"] or any(not isinstance(value, str) or not value.strip()
                                           for value in cfg["ocr"]["languages"]):
        raise ValueError("ocr.languages must contain language codes")
    if any(type(value) is not int or value not in (90, 180, 270) for value in cfg["ocr"]["rotation_info"]):
        raise ValueError("ocr.rotation_info supports 90, 180 and 270")
    if cfg["ocr"]["tile_size"] and cfg["ocr"]["tile_overlap"] >= cfg["ocr"]["tile_size"]:
        raise ValueError("ocr.tile_overlap must be smaller than tile_size")
    if cfg["preprocess"]["max_scale"] < 1 or cfg["vectorize"]["bezier_samples"] < 2:
        raise ValueError("max_scale must be >= 1; bezier_samples must be >= 2")
    if cfg["vectorize"]["binarize_block_size"] < 3:
        raise ValueError("vectorize.binarize_block_size must be >= 3")
    if cfg["classify"]["heuristic"]["mid_photo_norm"] <= 0:
        raise ValueError("classify.heuristic.mid_photo_norm must be positive")
    for key in ("text_in_table_type", "text_in_drawing_type"):
        if cfg["layout"][key] not in {"text", "annotation", "dimension"}:
            raise ValueError(f"Invalid layout.{key}")
    heuristic = cfg["classify"]["heuristic"]
    if not 0 <= heuristic["dark_gray_max"] < heuristic["light_gray_min"] <= 255:
        raise ValueError("Gray thresholds must satisfy 0 <= dark < light <= 255")


def load_config(path: str | Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, encoding="utf-8") as stream:
            user = json.load(stream)
        if not isinstance(user, dict):
            raise ValueError("Config must be an object")
        cfg = _merge(cfg, user)
    validate_config(cfg)
    return cfg


def resolve_config_path(path: str | Path | None = None, source_root: Path | None = None) -> Path | None:
    if path is not None:
        return Path(path)
    candidates = [Path.cwd() / "config.json"]
    if source_root is not None:
        candidates.append(source_root / "config.json")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def preflight(cfg: dict, files: list[Path]) -> tuple[dict, list[str]]:
    """Check local prerequisites without model downloads or network calls."""
    validate_config(cfg)
    effective = copy.deepcopy(cfg)
    warnings = []
    if not files:
        raise ValueError("No input files found")
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}:
            raise ValueError(f"Unsupported input: {source.name}")
    if importlib.util.find_spec("easyocr") is None:
        raise RuntimeError("EasyOCR is not installed")
    if effective["ocr"]["gpu"]:
        import torch
        if not torch.cuda.is_available():
            if not effective["ocr"]["cpu_fallback"]:
                raise RuntimeError("CUDA unavailable; enable ocr.cpu_fallback or disable ocr.gpu")
            effective["ocr"]["gpu"] = False
            warnings.append("CUDA unavailable; using CPU OCR")
    cache = Path(os.environ.get("EASYOCR_MODULE_PATH") or os.environ.get("MODULE_PATH")
                 or (Path.home() / ".EasyOCR")) / "model"
    if not cache.exists() or not any(cache.glob("*.pth")):
        warnings.append("OCR model cache missing; first run may download models")
    if effective["classify"]["use_vlm"]:
        provider = effective["classify"]["provider"]
        module = {"ollama": "ollama", "openai": "openai", "gemini": "google.genai"}[provider]
        try:
            installed = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            installed = False
        key_present = provider == "ollama" or bool(os.environ.get("OPENAI_API_KEY") if provider == "openai"
            else os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        if not installed or not key_present:
            effective["classify"]["use_vlm"] = False
            warnings.append(f"VLM {provider} unavailable; using heuristic")
        elif provider in {"openai", "gemini"}:
            warnings.append(f"Document crops will be sent to {provider}")
    return effective, warnings
