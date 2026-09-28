"""
tests/test_hierarchical_tiers.py — Verifica della segmentazione a 3 Tier e per franchise:
  - Box sigillati: Core (top 8, <500€), Bench (riserve), Vault (>=500€ o Grail), Avoid, Verify
  - Singole gradate: Core (top 8), Bench (posizioni 9+), Alternative
  - Etichettatura franchise (Pokémon EN, Pokémon JP, One Piece, Magic)
"""

import pytest
from scripts.generate_monthly_signal import compute_signal_rows
from scripts.generate_singles_signal import (
    compute_singles_signal_rows,
    compute_singles_alternative_rows,
    filter_singles_rows,
    PRODUCTION_PARAMS,
    DAC7_SINGLES_PARAMS,
)
from app import get_box_franchise_label


def test_box_signals_hierarchical_tiers():
    """Verifica che ogni box abbia un tier valido e che i tetti di classificazione siano rispettati."""
    rows, latest_date = compute_signal_rows()
    assert len(rows) > 0, "Nessun segnale box restituito"

    valid_tiers = {"core", "bench", "vault", "avoid", "verify", "excessive"}
    for r in rows:
        assert "tier" in r, f"Campo 'tier' mancante per {r.get('name')}"
        assert r["tier"] in valid_tiers, f"Tier non valido: {r['tier']} per {r.get('name')}"
        assert "franchise" in r, f"Campo 'franchise' mancante per {r.get('name')}"
        assert "language" in r, f"Campo 'language' mancante per {r.get('name')}"

        if r["tier"] == "vault":
            assert r["current_price_eur"] >= 500.0 or r.get("set_tier") == "Grail"

    # Verifica conteggio core per ciascun franchise (max 8 per franchise)
    core_rows = [r for r in rows if r["tier"] == "core"]
    assert len(core_rows) > 0, "Nessun box nel Tier 1 (Core)"

    by_franchise = {}
    for r in core_rows:
        key = (r["franchise"], r["language"])
        by_franchise[key] = by_franchise.get(key, 0) + 1
        assert by_franchise[key] <= 8, f"Più di 8 box Core per {key}: {by_franchise[key]}"


def test_box_franchise_label_helper():
    """Verifica la corretta risoluzione delle etichette franchise per i box."""
    assert get_box_franchise_label({"franchise": "pokemon", "language": "en"}) == "Pokémon EN"
    assert get_box_franchise_label({"franchise": "pokemon", "language": "jp"}) == "Pokémon JP"
    assert get_box_franchise_label({"franchise": "one_piece", "language": "en"}) == "One Piece TCG"
    assert get_box_franchise_label({"franchise": "magic", "language": "en"}) == "Magic (MTG)"


def test_singles_hierarchical_tiers_tagging():
    """Verifica che i segnali delle singole includano tier ('core', 'bench') e target_badge."""
    # Test su DAC7 params (20 carte)
    rows, latest_date = compute_singles_signal_rows(DAC7_SINGLES_PARAMS)
    assert len(rows) > 0

    core_rows = [r for r in rows if r.get("tier") == "core"]
    bench_rows = [r for r in rows if r.get("tier") == "bench"]

    assert len(core_rows) == min(8, len(rows)), f"Previste 8 carte Core, trovate {len(core_rows)}"
    if len(rows) > 8:
        assert len(bench_rows) == len(rows) - 8

    for r in rows:
        assert "target_badge" in r, f"Campo 'target_badge' mancante per {r.get('name')}"
        assert "PSA" in r["target_badge"], f"Badge anomalo: {r['target_badge']}"


def test_filter_singles_rows_retag_tiers():
    """Verifica che filter_singles_rows ri-etichetti correttamente i primi 8 come core quando retag_tiers=True."""
    dummy_rows = [
        {"item_id": f"card_{i}", "name": f"Card {i}", "current_price_eur": 50.0 + i,
         "rarity": "Secret Rare Holo", "franchise": "pokemon", "tier": "bench"}
        for i in range(15)
    ]
    filtered = filter_singles_rows(dummy_rows, min_price=40.0, max_price=500.0, retag_tiers=True)
    assert len(filtered) == 15
    assert sum(1 for r in filtered if r["tier"] == "core") == 8
    assert sum(1 for r in filtered if r["tier"] == "bench") == 7

    # Con retag_tiers=False mantiene il tier esistente
    filtered_no_retag = filter_singles_rows(dummy_rows, min_price=40.0, max_price=500.0, retag_tiers=False)
    assert all(r["tier"] == "bench" for r in filtered_no_retag)
