"""
poke_quant/data/fx_rates.py — Tassi di cambio EUR/USD storici REALI (Twelve Data),
in sostituzione della costante fissa DEFAULT_EUR_USD=1.08 usata finora per convertire
OGNI prezzo USD->EUR su tutto il periodo 2021-2026.

Il tasso reale è oscillato da 1.22 (2021) a 0.97 (fine 2022) a 1.19 (2026): usare una
costante fissa introduce un errore di conversione sistematico fino al 12% in singoli
mesi, in entrambe le direzioni, su OGNI serie storica costruita da price_fetcher.py.

data_cache/eur_usd_fx.csv contiene la serie mensile scaricata da Twelve Data
(EUR/USD, close mensile) da gennaio 2021 al mese più recente disponibile.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd

from poke_quant.data.storage import ensure_cache_dir

FX_FILENAME = "eur_usd_fx.csv"


def load_eur_usd_series() -> Optional[pd.Series]:
    path = ensure_cache_dir() / FX_FILENAME
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df["eur_usd"]


def rate_for_month(eur_usd_series: pd.Series, date: pd.Timestamp, default: float) -> float:
    """Tasso del mese richiesto; se il mese è fuori dalla serie (troppo vecchio o
    troppo recente), usa il tasso più vicino disponibile invece di un default
    arbitrario — solo se la serie è completamente assente si usa `default`."""
    if eur_usd_series is None or eur_usd_series.empty:
        return default
    month_start = pd.Timestamp(date.year, date.month, 1)
    if month_start in eur_usd_series.index:
        return float(eur_usd_series.loc[month_start])
    # Fuori range: usa il valore più vicino (nearest), non un default fisso.
    idx_pos = eur_usd_series.index.searchsorted(month_start)
    idx_pos = min(max(idx_pos, 0), len(eur_usd_series) - 1)
    return float(eur_usd_series.iloc[idx_pos])
