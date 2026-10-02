"""
tests/test_discover_universe_resilience.py — Regressione per il fallimento
reale del workflow mensile GitHub Action (2026-10-02, run 37025114175):
scripts/discover_sealed_universe.py e scripts/discover_pokemon_chinese_
sealed_universe.py crashavano con un traceback non gestito quando una
singola chiamata di rete upstream (pokemontcg.io/PriceCharting) rispondeva
con un errore (500/429) - riprodotto dal vivo lo stesso giorno (pokemontcg.io
rispondeva realmente 500 in quel momento). Entrambi gli script devono invece
fallire in modo pulito (messaggio chiaro, uscita non-zero SENZA traceback),
coerente con come il resto del motore dati tollera i fallimenti di rete
upstream (es. rebuild_prices_with_real_fx.py, che logga e continua).
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.discover_sealed_universe as discover_sealed_universe
import scripts.discover_pokemon_chinese_sealed_universe as discover_chinese


def test_discover_sealed_universe_fails_cleanly_on_upstream_500(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["discover_sealed_universe.py", "--dry-run"])

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "500 Server Error: Internal Server Error for url: https://api.pokemontcg.io/v2/sets"
    )
    with patch.object(discover_sealed_universe.requests, "get", return_value=mock_resp):
        with pytest.raises(SystemExit) as exc_info:
            discover_sealed_universe.main()

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "pokemontcg.io" in out
    assert "[ERRORE]" in out


def test_discover_pokemon_chinese_fails_cleanly_on_upstream_429(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["discover_pokemon_chinese_sealed_universe.py", "--dry-run"])

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "429 Client Error: Too Many Requests for url: https://www.pricecharting.com/category/pokemon-cards"
    )
    with patch.object(discover_chinese.requests, "get", return_value=mock_resp):
        with pytest.raises(SystemExit) as exc_info:
            discover_chinese.fetch_chinese_game_slugs()

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "PriceCharting" in out
    assert "[ERRORE]" in out


def test_discover_sealed_universe_still_works_on_success(monkeypatch):
    """Non regredire il percorso normale: una risposta 200 valida deve
    ancora funzionare esattamente come prima del fix."""
    monkeypatch.setattr(sys, "argv", ["discover_sealed_universe.py", "--dry-run"])

    sets_resp = MagicMock()
    sets_resp.raise_for_status.return_value = None
    sets_resp.json.return_value = {"data": []}
    with patch.object(discover_sealed_universe.requests, "get", return_value=sets_resp):
        discover_sealed_universe.main()  # non deve lanciare SystemExit
