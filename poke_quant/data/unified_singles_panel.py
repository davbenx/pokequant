"""
poke_quant/data/unified_singles_panel.py — Pannello prezzo UNIFICATO per le
singole multi-franchise (progetto pilota Magic: The Gathering, richiesto
dall'utente: "aggiungiamo... magic the gathering... integrato in una sola
strategia").

Pokemon/One Piece: il fattore di scarsita' opera su Grade 9 (slab gia'
gradato). Magic: The Gathering (scelta confermata dall'utente dopo verifica
che la gradazione MTG e' quasi assente fuori da poche carte vintage - vedi
scripts/discover_mtg_chase_singles.py): opera su RAW/ungraded. Per una
strategia REALMENTE unica (una sola regressione cross-sezionale su tutte le
franchise insieme, non tre pipeline separate), serve UN pannello dove ogni
colonna e' "il prezzo su cui quella carta viene davvero valutata" - qui, non
sparsa su due file diversi con semantiche diverse.

Regola: preferisce il pannello Grade 9 quando la carta ha dati reali li' (ogni
carta Pokemon/One Piece con mercato gradato), altrimenti usa il pannello RAW
(ogni carta MTG, e qualunque altra carta senza mercato gradato liquido) - non
una regola per franchise scritta a mano, generalizza correttamente a qualunque
carta futura senza dati gradati, franchise a parte.
"""

from __future__ import annotations
import pandas as pd


def build_unified_singles_price_panel(grade9_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Unisce grade9_df e raw_df in un solo pannello: per ogni item_id, usa la
    colonna di grade9_df se esiste e ha almeno un dato reale, altrimenti quella
    di raw_df. Le date sono l'unione degli indici dei due pannelli."""
    grade9_cols = set(grade9_df.columns) if grade9_df is not None else set()
    raw_cols = set(raw_df.columns) if raw_df is not None else set()

    idx = pd.Index([])
    if grade9_df is not None:
        idx = idx.union(grade9_df.index)
    if raw_df is not None:
        idx = idx.union(raw_df.index)
    idx = idx.sort_values()

    out = {}
    for item_id in grade9_cols | raw_cols:
        if item_id in grade9_cols and grade9_df[item_id].notna().any():
            out[item_id] = grade9_df[item_id]
        elif item_id in raw_cols:
            out[item_id] = raw_df[item_id]
    result = pd.DataFrame(out)
    return result.reindex(idx)
