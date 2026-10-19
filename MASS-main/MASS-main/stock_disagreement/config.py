from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("MASS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATASET_DIR = Path(os.environ.get("MASS_DATASET_DIR", PROJECT_ROOT / "stock_disagreement" / "dataset"))
RESULT_DIR = Path(os.environ.get("MASS_RESULT_DIR", PROJECT_ROOT / "stock_disagreement" / "res"))
LLM_BASE_URL = os.environ.get("MASS_LLM_BASE_URL", "https://models.sjtu.edu.cn/api/v1")
CHAT_MODEL_NAME = os.environ.get("MASS_CHAT_MODEL", os.environ.get("MASS_LLM_MODEL_NAME", "deepseek-chat"))
REASONING_MODEL_NAME = os.environ.get("MASS_REASONING_MODEL", "deepseek-reasoner")
VISION_MODEL_NAME = os.environ.get("MASS_VISION_MODEL", "qwen")
LLM_MODEL_NAME = CHAT_MODEL_NAME
LLM_API_KEY = os.environ.get("MASS_LLM_API_KEY", "")
LLM_MAX_TOKENS = int(os.environ.get("MASS_LLM_MAX_TOKENS", "4096"))
LLM_INPUT_COST_PER_MILLION = float(os.environ.get("MASS_LLM_INPUT_COST_PER_MILLION", "0"))
LLM_OUTPUT_COST_PER_MILLION = float(os.environ.get("MASS_LLM_OUTPUT_COST_PER_MILLION", "0"))
LLM_REQUESTS_PER_MINUTE = int(os.environ.get("MASS_LLM_REQUESTS_PER_MINUTE", "10"))
LLM_TOKENS_PER_MINUTE = int(os.environ.get("MASS_LLM_TOKENS_PER_MINUTE", "300000"))
LLM_TOKENS_PER_WEEK = int(os.environ.get("MASS_LLM_TOKENS_PER_WEEK", "1000000000"))
LLM_CACHE_MODE = os.environ.get("MASS_LLM_CACHE_MODE", "readwrite").lower()
LLM_CACHE_DIR = Path(os.environ.get("MASS_LLM_CACHE_DIR", RESULT_DIR / "llm_cache"))
_DATASET_VARIANT: str | None = None


def data_path(*parts: str) -> Path:
    return DATASET_DIR.joinpath(*parts)


def set_dataset_variant(variant: str | None) -> None:
    """Prefer variant-specific files while retaining root-level fallbacks."""
    global _DATASET_VARIANT
    _DATASET_VARIANT = variant.strip() if variant else None


def result_path(*parts: str) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR.joinpath(*parts)


def first_existing_data_path(*names: str) -> Path:
    if _DATASET_VARIANT:
        for name in names:
            path = data_path(_DATASET_VARIANT, name)
            if path.exists() and path.stat().st_size > 8:
                return path
    for name in names:
        path = data_path(name)
        if path.exists() and path.stat().st_size > 8:
            return path
    joined = ", ".join(names)
    raise FileNotFoundError(f"None of these dataset files exist or contain valid data: {joined}")
