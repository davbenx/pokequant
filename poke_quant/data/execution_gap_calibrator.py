"""
poke_quant/data/execution_gap_calibrator.py — Calibrazione empirica dello scarto tra
prezzo dashboard (PriceCharting/Cardmarket "trend"/mediano) e prezzo realmente eseguibile
(lowest active ask verificato manualmente al momento del check).

Motivazione: la friction attuale (config.PLATFORM_FEES + items_metadata.slippage_pct)
usa uno slippage flat 1-2%. Per prodotti out-of-print con disponibilità intermittente,
lo scarto reale osservato tra prezzo dashboard e primo ask disponibile può essere molto
più ampio e dipendente da tier/scarsità. Questo modulo NON inventa quello scarto: lo
stima solo da osservazioni loggate manualmente via scripts/log_execution_price.py.

Finché il log è vuoto o troppo piccolo, get_calibrated_slippage() ritorna esplicitamente
is_calibrated=False e il default statico — mai un numero fittizio spacciato per calibrato.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from poke_quant.data.storage import ensure_cache_dir

LOG_FILENAME = "execution_price_log.csv"
LOG_COLUMNS = [
    "date", "item_id", "dashboard_price_eur", "verified_lowest_ask_eur",
    "source", "active_listing_count", "notes",
]

MIN_OBS_FOR_CALIBRATION = 8  # sotto questa soglia non ci si fida della media empirica


def load_execution_log() -> Optional[pd.DataFrame]:
    path = ensure_cache_dir() / LOG_FILENAME
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df["premium_pct"] = (
        (df["verified_lowest_ask_eur"] - df["dashboard_price_eur"]) / df["dashboard_price_eur"]
    ) * 100.0
    return df


def append_execution_observation(
    item_id: str,
    dashboard_price_eur: float,
    verified_lowest_ask_eur: float,
    source: str = "cardmarket",
    active_listing_count: int = 1,
    notes: str = "",
    date: Optional[str] = None,
) -> Dict[str, Any]:
    path = ensure_cache_dir() / LOG_FILENAME
    row = {
        "date": date or pd.Timestamp.today().strftime("%Y-%m-%d"),
        "item_id": item_id,
        "dashboard_price_eur": dashboard_price_eur,
        "verified_lowest_ask_eur": verified_lowest_ask_eur,
        "source": source,
        "active_listing_count": active_listing_count,
        "notes": notes,
    }
    df_row = pd.DataFrame([row], columns=LOG_COLUMNS)
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)

    premium_pct = ((verified_lowest_ask_eur - dashboard_price_eur) / dashboard_price_eur) * 100.0 \
        if dashboard_price_eur > 0 else 0.0
    row["premium_pct"] = round(premium_pct, 2)
    return row


def compute_premium_stats_by_tier(
    log_df: pd.DataFrame, metadata: Dict[str, Any]
) -> pd.DataFrame:
    """Aggrega lo scarto osservato per set_tier. Ogni riga con n < MIN_OBS_FOR_CALIBRATION
    va letta come indicativa, non come stima affidabile."""
    tiers = log_df["item_id"].map(lambda k: metadata.get(k, {}).get("set_tier", "unknown"))
    tmp = log_df.assign(set_tier=tiers)
    agg = tmp.groupby("set_tier")["premium_pct"].agg(["count", "mean", "median", "std"])
    agg = agg.rename(columns={"count": "n_obs"})
    return agg.reset_index()


def get_calibrated_slippage(
    item_id: str,
    metadata: Dict[str, Any],
    log_df: Optional[pd.DataFrame],
    fallback_slippage_pct: Optional[float] = None,
    min_obs: int = MIN_OBS_FOR_CALIBRATION,
) -> Tuple[float, bool, int]:
    """
    Ritorna (slippage_pct_da_usare, is_calibrated, n_obs_usate).

    Priorità:
      1. Media empirica per lo stesso item_id, se n_obs >= min_obs.
      2. Media empirica per lo stesso set_tier, se n_obs >= min_obs.
      3. Fallback statico (config/metadata attuale) con is_calibrated=False.

    Non estrapola mai un numero "calibrato" da meno di min_obs osservazioni:
    è il punto centrale di questo modulo rispetto alla pratica precedente
    (slippage_pct fisso 1-2% dichiarato senza alcun dato a supporto).
    """
    default = fallback_slippage_pct
    if default is None:
        default = metadata.get(item_id, {}).get("slippage_pct", 0.02)

    if log_df is None or log_df.empty:
        return default, False, 0

    item_rows = log_df[log_df["item_id"] == item_id]
    if len(item_rows) >= min_obs:
        return float(item_rows["premium_pct"].mean() / 100.0), True, len(item_rows)

    tier = metadata.get(item_id, {}).get("set_tier")
    if tier is not None:
        tier_ids = {k for k, v in metadata.items() if v.get("set_tier") == tier}
        tier_rows = log_df[log_df["item_id"].isin(tier_ids)]
        if len(tier_rows) >= min_obs:
            return float(tier_rows["premium_pct"].mean() / 100.0), True, len(tier_rows)

    return default, False, len(item_rows)
