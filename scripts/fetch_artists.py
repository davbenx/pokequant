"""
scripts/fetch_artists.py — Arricchisce items_metadata.json con il campo "artist"
(illustratore) per ogni singola reale, leggendo i dati carta-per-carta da
pokemontcg.io (stessa fonte/stesso metodo di scripts/discover_chase_cards.py,
che non porta mai l'illustratore nel metadata salvato).

Resumable: salva un checkpoint per-set in data_cache/artists_done_sets.json,
così un'interruzione a metà (rete, timeout) non perde il lavoro già fatto -
un rerun salta i set già completati invece di richiamare pokemontcg.io da capo.

Usato per costruire e testare il fattore "illustratore" (field_name="artist" su
RarityTierFactorStrategy, vedi poke_quant/engine/strategies/rarity_tier_factor.py)
- risultato: NON VALIDATO (vedi docstring di quel file).

Uso:
    python scripts/fetch_artists.py
"""

import sys
import json
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, save_metadata, ensure_cache_dir
from scripts.discover_chase_cards import SET_IDS, fetch_set_cards, slugify_card

STATE_FILE = str(Path(ensure_cache_dir()) / "artists_done_sets.json")


def main():
    metadata = load_metadata()
    by_key = {}
    for item_id, v in metadata.items():
        if v.get("type") == "single" and v.get("game_slug") and v.get("item_slug"):
            by_key[(v["game_slug"], v["item_slug"])] = item_id

    done_sets = set()
    if os.path.exists(STATE_FILE):
        done_sets = set(json.load(open(STATE_FILE)))
    print(f"Singole da arricchire: {len(by_key)} | Set gia' completati in run precedenti: {len(done_sets)}", flush=True)

    n_already = sum(1 for v in metadata.values() if v.get("type") == "single" and v.get("artist"))
    for set_id, era in SET_IDS.items():
        if set_id in done_sets:
            continue
        cards = fetch_set_cards(set_id)
        print(f"{set_id} ({era}): {len(cards)} carte", flush=True)
        for c in cards:
            item_slug = slugify_card(c["name"], c["number"])
            game_slug = None
            for item_id, v in metadata.items():
                if v.get("type") == "sealed" and item_id.startswith(era) and v.get("game_slug"):
                    game_slug = v["game_slug"]
                    break
            if game_slug is None:
                continue
            key = (game_slug, item_slug)
            if key in by_key and c.get("artist"):
                metadata[by_key[key]]["artist"] = c["artist"]
        done_sets.add(set_id)
        save_metadata(metadata)
        json.dump(sorted(done_sets), open(STATE_FILE, "w"))
        print(f"  [checkpoint] set {set_id} completato e salvato ({len(done_sets)}/{len(SET_IDS)} totali)", flush=True)
        time.sleep(0.5)

    n_now = sum(1 for v in metadata.values() if v.get("type") == "single" and v.get("artist"))
    print(f"\nCompletato. Illustratori totali in metadata: {n_now} (+{n_now - n_already} in questa run)", flush=True)


if __name__ == "__main__":
    main()
