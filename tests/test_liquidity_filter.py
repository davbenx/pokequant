"""
tests/test_liquidity_filter.py — Unit test per il filtro di attendibilità/liquidità.
"""

import pandas as pd

from poke_quant.data.liquidity_filter import (
    compute_reliability_flags, filter_reliable, is_liquid_sealed, liquid_sealed_ids,
    compute_grade_raw_ratio_flags, liquid_singles_ids,
)


def _make_df():
    idx = pd.date_range("2021-01-01", periods=12, freq="MS")
    stable = pd.Series([100 + i for i in range(12)], index=idx)  # crescita lineare, ok
    thin_market = pd.Series([100] * 5 + [13000] + [100] * 6, index=idx)  # salto assurdo
    too_short = pd.Series([100, 110, 120], index=idx[:3])
    return pd.DataFrame({"stable_item": stable, "thin_item": thin_market, "short_item": too_short})


def test_stable_series_marked_reliable():
    flags = compute_reliability_flags(_make_df())
    ok, reason = flags["stable_item"]
    assert ok is True
    assert reason == ""


def test_extreme_monthly_jump_flagged_unreliable():
    flags = compute_reliability_flags(_make_df())
    ok, reason = flags["thin_item"]
    assert ok is False
    assert "salto mensile" in reason


def test_too_short_series_flagged_unreliable():
    flags = compute_reliability_flags(_make_df())
    ok, reason = flags["short_item"]
    assert ok is False
    assert "troppo corta" in reason


def test_filter_reliable_keeps_only_stable_column():
    filtered = filter_reliable(_make_df())
    assert list(filtered.columns) == ["stable_item"]


def _sealed_prices():
    idx = pd.date_range("2021-01-01", periods=6, freq="MS")
    return pd.DataFrame({
        "modern_bb": [100.0] * 6,
        "vintage_liquid_bb": [500.0] * 6,   # 5x MSRP 100 - dentro range
        "vintage_grail_bb": [5000.0] * 6,   # 50x MSRP 100 - fuori range (grail)
        "vintage_no_msrp_bb": [500.0] * 6,  # nessun MSRP - non fabbricato, escluso
    })


def _sealed_meta():
    return {
        "modern_bb": {"type": "sealed", "release_date": "2021-01-01", "msrp": 100.0},
        "vintage_liquid_bb": {"type": "sealed", "release_date": "2010-01-01", "msrp": 100.0},
        "vintage_grail_bb": {"type": "sealed", "release_date": "2005-01-01", "msrp": 100.0},
        "vintage_no_msrp_bb": {"type": "sealed", "release_date": "2008-01-01", "msrp": None},
    }


def test_modern_box_always_included_regardless_of_ratio():
    meta, prices = _sealed_meta(), _sealed_prices()
    assert is_liquid_sealed("modern_bb", meta["modern_bb"], prices, max_price_msrp_ratio=10.0)


def test_vintage_box_included_if_price_msrp_ratio_within_range():
    meta, prices = _sealed_meta(), _sealed_prices()
    assert is_liquid_sealed("vintage_liquid_bb", meta["vintage_liquid_bb"], prices, max_price_msrp_ratio=10.0)


def test_vintage_grail_excluded_when_ratio_exceeds_range():
    meta, prices = _sealed_meta(), _sealed_prices()
    assert not is_liquid_sealed("vintage_grail_bb", meta["vintage_grail_bb"], prices, max_price_msrp_ratio=10.0)


def test_vintage_box_without_real_msrp_stays_excluded_not_fabricated():
    meta, prices = _sealed_meta(), _sealed_prices()
    assert not is_liquid_sealed("vintage_no_msrp_bb", meta["vintage_no_msrp_bb"], prices, max_price_msrp_ratio=10.0)


def test_liquid_sealed_ids_returns_expected_subset():
    metadata, prices = _sealed_meta(), _sealed_prices()
    ids = liquid_sealed_ids(metadata, prices, max_price_msrp_ratio=10.0)
    assert set(ids) == {"modern_bb", "vintage_liquid_bb"}


def _grade_raw_cohort(n_normal=25, outlier_ratio=1.0, normal_ratio=5.0):
    """Coorte di carte 1999 con rapporto grade9/raw normale (~normal_ratio), piu'
    una carta 'thin_outlier' con rapporto molto piu' basso (come raichu_14: serie
    liscia, ma persistentemente sottostimata vs. il prezzo raw di riferimento)."""
    idx = pd.date_range("2021-01-01", periods=6, freq="MS")
    metadata, prices = {}, {}
    for i in range(n_normal):
        item_id = f"normal_{i}"
        metadata[item_id] = {"type": "single", "release_date": "1999-01-01", "cardmarket_ref_price_eur": 50.0}
        prices[item_id] = [50.0 * normal_ratio] * 6
    metadata["thin_outlier"] = {"type": "single", "release_date": "1999-06-01", "cardmarket_ref_price_eur": 50.0}
    prices["thin_outlier"] = [50.0 * outlier_ratio] * 6
    return metadata, pd.DataFrame(prices, index=idx)


def test_thin_outlier_flagged_below_cohort_percentile():
    metadata, prices = _grade_raw_cohort()
    flags = compute_grade_raw_ratio_flags(metadata, prices, percentile_cutoff=0.10, min_cohort=20)
    assert "thin_outlier" in flags
    ok, reason = flags["thin_outlier"]
    assert ok is False
    assert "grade9/raw" in reason


def test_normal_cohort_members_not_flagged():
    metadata, prices = _grade_raw_cohort()
    flags = compute_grade_raw_ratio_flags(metadata, prices, percentile_cutoff=0.10, min_cohort=20)
    assert "normal_0" not in flags


def test_small_cohort_skipped_no_data_fabricated():
    """Sotto min_cohort, nessun giudizio - dato insufficiente, non si flagga alla cieca."""
    metadata, prices = _grade_raw_cohort(n_normal=5)
    flags = compute_grade_raw_ratio_flags(metadata, prices, percentile_cutoff=0.10, min_cohort=20)
    assert flags == {}


def test_missing_reference_price_not_flagged():
    metadata, prices = _grade_raw_cohort()
    del metadata["thin_outlier"]["cardmarket_ref_price_eur"]
    flags = compute_grade_raw_ratio_flags(metadata, prices, percentile_cutoff=0.10, min_cohort=20)
    assert "thin_outlier" not in flags


def _singles_universe():
    idx = pd.date_range("2024-01-01", periods=12, freq="MS")
    metadata = {
        "cheap_common": {"type": "single"},  # sotto il pavimento di costo di gradazione
        "normal_card": {"type": "single"},
        "flagged_thin": {"type": "single", "data_quality": "thin_unreliable"},
        "a_sealed_box": {"type": "sealed"},
    }
    prices = pd.DataFrame({
        "cheap_common": [11.0] * 12,   # mediana 11EUR - sotto un pavimento di 20EUR
        "normal_card": [50.0] * 12,
        "flagged_thin": [80.0] * 12,   # prezzo alto, ma gia' flaggato dati inattendibili
        "a_sealed_box": [200.0] * 12,
    }, index=idx)
    return metadata, prices


def test_liquid_singles_ids_excludes_thin_unreliable_flagged_cards():
    """Trovato indagando un prezzo reale (Mantine #64 a 11,48EUR Grade9): il
    segnale live (scripts/generate_singles_signal.py) non applicava MAI questo
    filtro, anche se il backtest validato lo fa da tempo - un bug, non solo un
    filtro mancante."""
    metadata, prices = _singles_universe()
    ids = liquid_singles_ids(metadata, prices, min_median_price_eur=0.0)
    assert "flagged_thin" not in ids
    assert "normal_card" in ids


def test_liquid_singles_ids_excludes_below_grading_cost_floor():
    """La sola gradazione PSA/CGC costa piu' del prezzo mostrato per queste
    carte - nessuno le gradirebbe oggi a queste condizioni (vedi
    scripts/grading_cost_floor_test.py)."""
    metadata, prices = _singles_universe()
    ids = liquid_singles_ids(metadata, prices, min_median_price_eur=20.0)
    assert "cheap_common" not in ids
    assert "normal_card" in ids


def test_liquid_singles_ids_excludes_sealed_items():
    metadata, prices = _singles_universe()
    ids = liquid_singles_ids(metadata, prices, min_median_price_eur=0.0)
    assert "a_sealed_box" not in ids


def test_liquid_singles_ids_uses_recent_price_not_stale_history():
    """BUG TROVATO (l'utente: "Vedo ancora Sandslash #42, che e' sotto il
    prezzo da gradazione"): la mediana su TUTTA la storia lasciava passare
    una carta scesa e rimasta bassa per mesi, perche' i prezzi vecchi piu'
    alti gonfiavano la mediana sopra soglia - Sandslash #42 (mediana storica
    20,32EUR, ma ~14EUR negli ultimi 7 mesi) ne era un caso reale. Deve
    restare escluso chi e' sceso e resta basso, mentre chi e' genuinamente
    risalito sopra soglia di recente (es. Lombre #34 nel caso reale) deve
    rientrare - la finestra recente conta, non la storia intera."""
    idx = pd.date_range("2024-01-01", periods=12, freq="MS")
    # "declined_zombie": valeva 25EUR per 9 mesi (mediana storica alta), poi
    # sceso a 14EUR negli ultimi 3 - la mediana storica lo farebbe passare,
    # quella recente no.
    declined_zombie = pd.Series([25.0] * 9 + [14.0] * 3, index=idx)
    # "recovered": simmetrico, era a 14EUR per 9 mesi, risalito a 25EUR negli
    # ultimi 3 - deve rientrare, non restare escluso per il suo passato.
    recovered = pd.Series([14.0] * 9 + [25.0] * 3, index=idx)
    prices = pd.DataFrame({"declined_zombie": declined_zombie, "recovered": recovered})
    metadata = {
        "declined_zombie": {"type": "single"},
        "recovered": {"type": "single"},
    }
    ids = liquid_singles_ids(metadata, prices, min_median_price_eur=20.0, price_window_months=3)
    assert "declined_zombie" not in ids
    assert "recovered" in ids


def test_liquid_singles_ids_catches_sudden_single_month_crash():
    """BUG TROVATO (l'utente, due carte reali: "Unown [K] #58" e "Dark Golduck
    #37" - "ancora prezzi che e' impossibile trovare gradate"): diverso dal caso
    Sandslash (calo SOSTENUTO mascherato dalla mediana su tutta la storia) - qui
    il calo e' improvviso e recentissimo (un solo mese), ma la mediana della
    finestra a 3 mesi resta sopra soglia perche' gli altri 2 mesi sono ancora
    "vecchi" e alti. Unown K #58 reale: [22.55, 22.38, 14.63] -> mediana 22.38
    (sopra soglia) ma prezzo ATTUALE 14.63 (sotto). Deve restare escluso finche'
    anche l'ultimo prezzo non supera la soglia, non solo la mediana della
    finestra."""
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    sudden_crash = pd.Series([22.55, 22.38, 14.63], index=idx)  # caso reale Unown K #58
    stable_above = pd.Series([22.0, 22.0, 22.0], index=idx)  # controllo: nessun crollo
    prices = pd.DataFrame({"sudden_crash": sudden_crash, "stable_above": stable_above})
    metadata = {
        "sudden_crash": {"type": "single"},
        "stable_above": {"type": "single"},
    }
    ids = liquid_singles_ids(metadata, prices, min_median_price_eur=20.0, price_window_months=3)
    assert "sudden_crash" not in ids
    assert "stable_above" in ids


def test_liquid_singles_ids_skips_grading_floor_for_magic_franchise():
    """Progetto pilota MTG: il pannello per le carte franchise="magic" e'
    RAW/ungraded (non Grade 9 come Pokemon/One Piece) - il pavimento di costo
    di gradazione non ha senso per una carta che non si intende gradare, e
    non deve escludere carte MTG economiche che altrimenti sarebbero
    legittimamente investibili sul mercato raw."""
    idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    cheap_mtg_raw = pd.Series([2.0, 2.0, 2.0], index=idx)
    cheap_pokemon = pd.Series([2.0, 2.0, 2.0], index=idx)
    prices = pd.DataFrame({"cheap_mtg_raw": cheap_mtg_raw, "cheap_pokemon": cheap_pokemon})
    metadata = {
        "cheap_mtg_raw": {"type": "single", "franchise": "magic"},
        "cheap_pokemon": {"type": "single", "franchise": "pokemon"},
    }
    ids = liquid_singles_ids(metadata, prices, min_median_price_eur=20.0, price_window_months=3)
    assert "cheap_mtg_raw" in ids
    assert "cheap_pokemon" not in ids
