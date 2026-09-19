"""
poke_quant/data/storage.py — Gestione del salvataggio e caricamento dei dati storici in cache locale.
Garantisce esecuzioni di backtest offline, deterministiche e istantanee.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def save_price_matrix(df: pd.DataFrame, filename: str = "historical_prices.csv"):
    path = ensure_cache_dir() / filename
    df.to_csv(path, index=True)


def load_price_matrix(filename: str = "historical_prices.csv") -> Optional[pd.DataFrame]:
    path = ensure_cache_dir() / filename
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def save_metadata(metadata: Dict[str, Any], filename: str = "items_metadata.json"):
    path = ensure_cache_dir() / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_metadata(filename: str = "items_metadata.json") -> Optional[Dict[str, Any]]:
    path = ensure_cache_dir() / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_macro_matrix(df: pd.DataFrame, filename: str = "macro_benchmarks.csv"):
    path = ensure_cache_dir() / filename
    df.to_csv(path, index=True)


def load_macro_matrix(filename: str = "macro_benchmarks.csv") -> Optional[pd.DataFrame]:
    path = ensure_cache_dir() / filename
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

