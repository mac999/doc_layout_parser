"""Load config.json and merge it over the built-in defaults."""
import json
from pathlib import Path

DEFAULTS = {
    "input_dir": "input",
    "output_dir": "output",
    "preprocess": {"upscale_target_min_side": 1400, "max_scale": 3.0},
    "pdf": {"render_dpi": 200, "use_native_text": True, "use_native_vectors": True, "min_native_words": 5},
    "ocr": {"languages": ["ko", "en"], "gpu": True, "min_confidence": 0.30, "paragraph": False,
            "mag_ratio": 1.0, "canvas_size": 3200, "min_size": 8,
            "text_threshold": 0.6, "low_text": 0.35},
    "layout": {"min_region_area": 3000, "text_mask_padding": 4, "dilate_kernel": 21, "max_page_cover_ratio": 1.0},
    "classify": {
        "use_vlm": False, "ambiguous_only": True, "heuristic_confidence_threshold": 0.75,
        "provider": "ollama", "ollama_model": "llava:latest",
        "openai_model": "gpt-4o-mini", "gemini_model": "gemini-2.0-flash", "timeout_sec": 60,
    },
    "vectorize": {
        "binarize_block_size": 35, "binarize_C": 11,
        "min_polyline_length_px": 8.0, "simplify_epsilon": 2.0, "bezier_samples": 8,
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


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        return _merge(DEFAULTS, user)
    return dict(DEFAULTS)
