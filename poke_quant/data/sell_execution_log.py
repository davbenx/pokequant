"""
poke_quant/data/sell_execution_log.py — Registro delle VENDITE reali, per
misurare quello che nessun backtest o log di solo acquisto puo' catturare:
se un'inserzione trova davvero un compratore, a quale prezzo REALIZZATO
(spesso sotto il prezzo di listino - contrattazione, sconto per vendita
rapida) e dopo quanti giorni. Motivazione (osservazione diretta dell'utente):
"l'unico modo per fare un test e' comprare e vendere davvero. Non posso
simulare il fatto che mi accettino una proposta di acquisto o a quanto e in
quanto tempo riuscirò a vendere" - execution_gap_calibrator.py cattura solo
lo scarto all'ACQUISTO (dashboard vs ask visto), non dice nulla sul lato
vendita: prezzo di listino vs realizzato, tempo sul mercato, o se non vende
affatto (withdrawn - dato censurato, importante quanto una vendita riuscita).

Tre stati per riga: "listed" (appena messo in vendita, esito ancora
sconosciuto), "sold" (venduto, con prezzo realizzato e giorni sul mercato),
"withdrawn" (ritirato senza vendere - NON e' un dato da scartare, e' un
"tempo di attesa troncato" che va nella statistica del sell-through rate).

Finché il log è vuoto o troppo piccolo, ogni funzione di aggregazione ritorna
esplicitamente dati insufficienti - mai un numero fittizio spacciato per
calibrato (stesso principio di execution_gap_calibrator.py).
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd

from poke_quant.data.storage import ensure_cache_dir

LOG_FILENAME = "sell_execution_log.csv"
LOG_COLUMNS = [
    "item_id", "list_date", "list_price_eur", "dashboard_price_at_listing_eur",
    "status", "sold_date", "realized_price_eur", "days_on_market", "notes",
]

MIN_OBS_FOR_STATS = 8  # stesso principio di execution_gap_calibrator.MIN_OBS_FOR_CALIBRATION


def load_sell_log() -> Optional[pd.DataFrame]:
    path = ensure_cache_dir() / LOG_FILENAME
    if not path.exists():
        return None
    # dtype=str su tutto e keep_default_na=False: una colonna con SOLO celle
    # vuote (es. "notes" quando nessuna riga ha ancora un commento) verrebbe
    # altrimenti inferita float64/NaN da pandas, e una scrittura successiva
    # di una stringa in quella colonna fallirebbe con TypeError.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if df.empty:
        return None
    df["list_date"] = pd.to_datetime(df["list_date"])
    df["sold_date"] = pd.to_datetime(df["sold_date"].replace("", pd.NaT))
    for col in ("list_price_eur", "dashboard_price_at_listing_eur", "realized_price_eur", "days_on_market"):
        df[col] = pd.to_numeric(df[col].replace("", None))
    return df


def _save(df: pd.DataFrame) -> None:
    path = ensure_cache_dir() / LOG_FILENAME
    df.to_csv(path, index=False)


def log_new_listing(
    item_id: str,
    list_price_eur: float,
    dashboard_price_eur: Optional[float] = None,
    list_date: Optional[str] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Registra una nuova inserzione, esito ancora sconosciuto."""
    row = {
        "item_id": item_id,
        "list_date": list_date or pd.Timestamp.today().strftime("%Y-%m-%d"),
        "list_price_eur": list_price_eur,
        "dashboard_price_at_listing_eur": dashboard_price_eur if dashboard_price_eur is not None else "",
        "status": "listed",
        "sold_date": "",
        "realized_price_eur": "",
        "days_on_market": "",
        "notes": notes,
    }
    df_row = pd.DataFrame([row], columns=LOG_COLUMNS)
    path = ensure_cache_dir() / LOG_FILENAME
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)
    return row


def _find_open_listing(df: pd.DataFrame, item_id: str, list_date: Optional[str]) -> Optional[int]:
    """Trova l'indice della riga 'listed' che corrisponde - per item_id, e per
    list_date se specificato (altrimenti la piu' recente ancora aperta per
    quell'item_id, comodo se non si e' annotata la data esatta)."""
    candidates = df[(df["item_id"] == item_id) & (df["status"] == "listed")]
    if list_date:
        candidates = candidates[candidates["list_date"] == pd.to_datetime(list_date)]
    if candidates.empty:
        return None
    return candidates.index[-1]


def log_sale_outcome(
    item_id: str,
    realized_price_eur: float,
    list_date: Optional[str] = None,
    sold_date: Optional[str] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Chiude un'inserzione 'listed' come venduta. Se non trova una riga
    aperta corrispondente (es. vendita immediata mai loggata come 'listed'),
    crea una riga gia' chiusa con list_date=sold_date (holding a 0 giorni,
    onesto: non si inventa un tempo sul mercato che non e' stato osservato)."""
    df = load_sell_log()
    sold_dt = pd.to_datetime(sold_date or pd.Timestamp.today().strftime("%Y-%m-%d"))

    if df is not None:
        idx = _find_open_listing(df, item_id, list_date)
        if idx is not None:
            df.loc[idx, "status"] = "sold"
            df.loc[idx, "sold_date"] = sold_dt
            df.loc[idx, "realized_price_eur"] = realized_price_eur
            df.loc[idx, "days_on_market"] = (sold_dt - df.loc[idx, "list_date"]).days
            if notes:
                df.loc[idx, "notes"] = notes
            _save(df)
            return df.loc[idx].to_dict()

    row = {
        "item_id": item_id,
        "list_date": pd.to_datetime(list_date) if list_date else sold_dt,
        "list_price_eur": realized_price_eur,
        "dashboard_price_at_listing_eur": "",
        "status": "sold",
        "sold_date": sold_dt,
        "realized_price_eur": realized_price_eur,
        "days_on_market": 0 if not list_date else (sold_dt - pd.to_datetime(list_date)).days,
        "notes": notes,
    }
    new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True) if df is not None else pd.DataFrame([row], columns=LOG_COLUMNS)
    _save(new_df)
    return row


def log_withdrawn(item_id: str, list_date: Optional[str] = None, notes: str = "") -> Dict[str, Any]:
    """Chiude un'inserzione senza vendita - dato censurato, non da scartare:
    entra nel tasso di successo (sell-through rate) come un fallimento."""
    df = load_sell_log()
    if df is None:
        raise ValueError(f"nessuna inserzione registrata per {item_id} - usa log_new_listing prima")
    idx = _find_open_listing(df, item_id, list_date)
    if idx is None:
        raise ValueError(f"nessuna inserzione APERTA trovata per {item_id}" + (f" alla data {list_date}" if list_date else ""))
    df.loc[idx, "status"] = "withdrawn"
    if notes:
        df.loc[idx, "notes"] = notes
    _save(df)
    return df.loc[idx].to_dict()


def compute_sell_stats(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Statistiche aggregate - ognuna None finché non ci sono abbastanza
    osservazioni chiuse (sold+withdrawn) per fidarsene, mai un numero
    fittizio spacciato per calibrato."""
    if df is None:
        df = load_sell_log()
    if df is None or df.empty:
        return {"n_listed": 0, "n_sold": 0, "n_withdrawn": 0, "sell_through_rate": None,
                "median_days_on_market": None, "median_discount_from_list_pct": None,
                "is_reliable": False}

    n_listed_open = int((df["status"] == "listed").sum())
    sold = df[df["status"] == "sold"].copy()
    withdrawn = df[df["status"] == "withdrawn"]
    n_sold, n_withdrawn = len(sold), len(withdrawn)
    n_closed = n_sold + n_withdrawn

    sell_through_rate = (n_sold / n_closed) if n_closed > 0 else None
    median_days = float(sold["days_on_market"].median()) if n_sold > 0 else None
    if n_sold > 0:
        discount_pct = (sold["realized_price_eur"].astype(float) - sold["list_price_eur"].astype(float)) \
            / sold["list_price_eur"].astype(float) * 100.0
        median_discount = float(discount_pct.median())
    else:
        median_discount = None

    return {
        "n_listed_open": n_listed_open, "n_sold": n_sold, "n_withdrawn": n_withdrawn,
        "sell_through_rate": sell_through_rate, "median_days_on_market": median_days,
        "median_discount_from_list_pct": median_discount,
        "is_reliable": n_closed >= MIN_OBS_FOR_STATS,
    }
