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
            "text_threshold": 0.6, "low_text": 0.35,
            "line_gap_factor": 1.5, "line_row_factor": 0.6,
            "short_line_max_tokens": 3, "dim_token_ratio": 0.5},
    "layout": {"min_region_area": 3000, "text_mask_padding": 4, "dilate_kernel": 21, "max_page_cover_ratio": 1.0,
               "page_border_margin_px": 8,
               "text_in_drawing_type": "annotation", "text_in_table_type": "text",
               "text_region_overlap_ratio": 0.5},
    "classify": {
        "use_vlm": False, "ambiguous_only": True, "heuristic_confidence_threshold": 0.75,
        "provider": "ollama", "ollama_model": "llava:latest",
        "openai_model": "gpt-4o-mini", "gemini_model": "gemini-2.0-flash", "timeout_sec": 60,
        "vlm_max_image_side": 768,
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


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        return _merge(DEFAULTS, user)
    return dict(DEFAULTS)
