"""
poke_quant/data/external_psa10_history.py — Riconciliazione con il dataset esterno
gratuito github.com/samaygodika/pokemon-psa10-history (popolazione PSA10/PSA9 e vendite,
scraper community su alt.xyz, aggiornato ogni notte, storico giornaliero reale dall'
11 settembre 2026).

Trovato rispondendo a "prova a cercare i dati delle popolazioni passate anche su
GitHub o altri database gratuiti" (2026-09-25). Verificato live: 66.344 asset, storico
giornaliero reale (non uno snapshot unico), accessibile senza API key via
raw.githubusercontent.com (i file history/assets.csv e history/daily/<data>.csv esistono
e sono stati scaricati e ispezionati, non solo annunciati nel README).

LIMITI DICHIARATI (prima di usare questo modulo):
1. Storico reale SOLO dall'11 settembre 2026 (~2 settimane più vecchio del nostro, non
   un archivio retroattivo 2021-2026 - nessuna fonte gratuita trovata lo è).
2. Progetto hobby di terzi (nessun file LICENSE nel repo, nessuna garanzia di
   continuità) - non un provider commerciale con SLA.
3. Copre solo PSA (population PSA10/PSA9) - non CGC/BGS/SGC come invece fa in parte
   PriceCharting (population_fetcher.py) - complementare, non sostitutivo.
4. Indicizzato per asset_id di alt.xyz, non per i nostri item_id - la riconciliazione
   qui sotto NON è perfetta: sul nostro universo singole (3.096 carte), il matcher
   risolve un candidato univoco per il 77% (2.384/3.096) dopo aver escluso le varianti
   in lingua estera (stesso numero+soggetto ricorre in spagnolo/tedesco/italiano/ecc.,
   ognuna un asset diverso con popolazione diversa - il primo bug trovato qui) e aver
   penalizzato promo/reverse-holo/esclusive quando la nostra carta non lo è. Il 23%
   restante (712 carte) resta escluso per ambiguità reale (stesso numero+soggetto in
   PIÙ set/epoche completamente diversi nel catalogo alt.xyz di 65k carte, non risolvibile
   senza un ID esterno comune) - meglio pochi match precisi che molti match sbagliati.
"""

from __future__ import annotations
import re
from typing import Dict, List, Any, Optional

LANGUAGE_WORDS = {
    "spanish", "german", "italian", "portuguese", "french", "japanese", "korean",
    "chinese", "dutch", "polish", "russian", "thai", "indonesian", "vietnamese",
    "turkish", "arabic",
}
DEPRIORITIZE_WORDS = {"reverse", "promo", "exclusive", "retailer", "trick", "trade", "bonus", "stamp", "confetti"}


def normalize(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _is_foreign_language_variant(asset_row: Dict[str, str]) -> bool:
    """Lo stesso numero+soggetto ricorre in più lingue nel dataset (asset diversi,
    popolazione diversa) - la lingua compare come parola in 'variety' (es. 'German')
    o in coda a 'set' (es. 'Evolving Skies Spanish'), mai in un campo lingua dedicato."""
    words = set(normalize(asset_row.get("variety", "")).split()) | set(normalize(asset_row.get("set", "")).split())
    return bool(words & LANGUAGE_WORDS)


def _set_words_from_game_slug(game_slug: str) -> set:
    s = (game_slug or "").replace("pokemon-", "").replace("-", " ")
    return set(normalize(s).split())


def _match_penalty(asset_row: Dict[str, str], our_rarity_norm: str) -> tuple:
    """Punteggio piu' basso = candidato migliore. Penalizza varianti promo/reverse-holo/
    esclusive quando la NOSTRA carta non lo e' (evita di agganciare per sbaglio una
    popolazione di una stampa diversa dalla nostra), poi preferisce il nome piu' corto
    (le stampe 'semplici' hanno tipicamente un card_name piu' breve delle esclusive)."""
    words = set(normalize(asset_row.get("card_name", "")).split())
    penalty = len(words & DEPRIORITIZE_WORDS)
    if "reverse" in words and "reverse" not in our_rarity_norm:
        penalty += 5
    return (penalty, len(asset_row.get("card_name", "")))


def find_matching_asset(
    our_name: str,
    our_game_slug: str,
    our_rarity: Optional[str],
    number: str,
    candidates_by_number: Dict[str, List[Dict[str, str]]],
) -> Optional[Dict[str, str]]:
    """Trova l'asset alt.xyz corrispondente a una nostra carta (numero+set+nome), o None
    se nessun candidato univoco emerge dopo i filtri. candidates_by_number e' l'indice
    {card_number: [righe di assets.csv]} - costruito una volta dal chiamante per tutte
    le carte, non ricostruito ad ogni chiamata."""
    name_norm = normalize(our_name)
    rarity_norm = normalize(our_rarity or "")
    gs_words = _set_words_from_game_slug(our_game_slug)
    candidates = candidates_by_number.get(number.upper(), [])

    hits = []
    for row in candidates:
        if _is_foreign_language_variant(row):
            continue
        subj_norm = normalize(row.get("subject", ""))
        if not subj_norm:
            continue
        subj_ok = subj_norm in name_norm or name_norm.split(" ")[0] == subj_norm.split(" ")[0]
        set_norm_words = set(normalize(row.get("set", "")).split())
        set_ok = bool(gs_words) and gs_words.issubset(set_norm_words)
        if subj_ok and set_ok:
            hits.append(row)

    if not hits:
        return None
    hits.sort(key=lambda r: _match_penalty(r, rarity_norm))
    return hits[0]


def build_number_index(asset_rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = {}
    for row in asset_rows:
        num = (row.get("card_number") or "").strip().upper()
        if num:
            index.setdefault(num, []).append(row)
    return index


def extract_number_from_item_slug(item_slug: str) -> Optional[str]:
    m = re.search(r"-([a-zA-Z0-9]+)$", item_slug or "")
    return m.group(1).upper() if m else None
