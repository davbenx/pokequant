"""
poke_quant/data/liquidity_filter.py — Filtro di attendibilità/liquidità sulle serie prezzo.

Trovato espandendo l'universo a 389 asset: molti box sigillati vintage (e alcune
singole ultra-rare/promo) hanno volumi di vendita reali così bassi che il prezzo
guida di PriceCharting NON è rumore accettabile ma dato inutilizzabile — es.
neo_destiny_bb passa da ~13.000€ a 53,7€ in un mese, skyridge_bb tocca 182.915€.
Non è un dato "reale ma volatile": è un artefatto di mercato troppo sottile per
avere un prezzo mensile significativo (spesso una sola vendita anomala per mese).

Questo modulo NON cancella quei dati (restano in historical_prices.csv per
trasparenza) — li FLAGGA, cosicché ogni script di validazione possa escluderli
esplicitamente dall'universo investibile invece di lasciare che dominino
silenziosamente i risultati di una strategia.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd

DEFAULT_MAX_MONTHLY_JUMP = 2.0   # +-200% in un mese singolo
DEFAULT_MAX_RANGE_RATIO = 15.0   # max/min sull'intera serie


def compute_reliability_flags(
    prices_df: pd.DataFrame,
    max_monthly_jump: float = DEFAULT_MAX_MONTHLY_JUMP,
    max_range_ratio: float = DEFAULT_MAX_RANGE_RATIO,
    min_observations: int = 6,
) -> Dict[str, Tuple[bool, str]]:
    """Ritorna {item_id: (is_reliable, motivo)} per ogni colonna di prices_df."""
    flags: Dict[str, Tuple[bool, str]] = {}
    for col in prices_df.columns:
        s = prices_df[col].dropna()
        s = s[s > 0]
        if len(s) < min_observations:
            flags[col] = (False, f"serie troppo corta ({len(s)} osservazioni)")
            continue
        mom_ret = s.pct_change().dropna()
        max_jump = float(mom_ret.abs().max()) if not mom_ret.empty else 0.0
        ratio = float(s.max() / s.min())
        if max_jump > max_monthly_jump:
            flags[col] = (False, f"salto mensile {max_jump*100:.0f}% (soglia {max_monthly_jump*100:.0f}%)")
        elif ratio > max_range_ratio:
            flags[col] = (False, f"range max/min {ratio:.1f}x (soglia {max_range_ratio:.1f}x)")
        else:
            flags[col] = (True, "")
    return flags


def get_reliable_columns(prices_df: pd.DataFrame, **kwargs) -> List[str]:
    flags = compute_reliability_flags(prices_df, **kwargs)
    return [col for col, (ok, _) in flags.items() if ok]


def filter_reliable(prices_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Ritorna prices_df con solo le colonne che passano il filtro di attendibilità."""
    return prices_df[get_reliable_columns(prices_df, **kwargs)]
