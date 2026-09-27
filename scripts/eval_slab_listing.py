#!/usr/bin/env python3
"""
scripts/eval_slab_listing.py — Tool rapido da riga di comando per valutare
qualunque inserzione di carta gradata (PSA, BGS, CGC, SGC, GRAAD, PCA, ACE, ecc.).

Uso:
  python scripts/eval_slab_listing.py --card "kabutops" --company CGC --grade 9 --price 85 --shipping 6
  python scripts/eval_slab_listing.py --card "azumarill 114" --company PCA --grade 9 --price 90 --shipping 12
  python scripts/eval_slab_listing.py --card "dark slowbro" --company BGS --grade 9.5 --price 105
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.slabs.grading_multipliers import (
    GradingCompany,
    Era,
    normalize_company,
    normalize_era,
    get_grading_adjustment,
    adjust_price_for_grading,
    get_variant_multiplier,
    variant_to_pricecharting_key,
    SPECIAL_VARIANTS,
)
from poke_quant.config import estimate_usa_import_landed_cost, IMPORT_FROM_USA
from poke_quant.data.price_fetcher import fetch_pricecharting_variant_grade9
from scripts.generate_singles_signal import (
    _liquid_universe,
    _snapshot_for_date,
    PRODUCTION_PARAMS,
    PRESERVE_EDGE_ALPHA,
)
from poke_quant.engine.strategies.scarcity_value_factor import ScarcityValueFactorStrategy


def find_best_card_match(query: str, metadata: dict, prices_full: pd.DataFrame):
    """Cerca la corrispondenza più probabile tra le carte investibili."""
    q = query.strip().lower()
    matches = []
    for item_id, info in metadata.items():
        name = info.get("name", "").lower()
        slug = info.get("item_slug", "").lower()
        game = info.get("game_slug", "").lower()
        if q in name or q in item_id.lower() or q in slug:
            matches.append((item_id, info))
    
    if not matches:
        # Ricerca per parole separate (es. "azumarill 114")
        words = q.split()
        for item_id, info in metadata.items():
            combined = f"{info.get('name', '')} {item_id} {info.get('game_slug', '')}".lower()
            if all(w in combined for w in words):
                matches.append((item_id, info))

    return matches


def evaluate_listing(
    card_query: str,
    company: str,
    grade: str | float,
    price_eur: float,
    shipping_eur: float = 0.0,
    era_override: str = None,
    is_pristine: bool = False,
    is_black_label: bool = False,
    variant: str = "standard",
    is_usa: bool = False,
):
    metadata = load_metadata()
    prices_full = load_price_matrix("historical_prices_graded_singles_grade9.csv")
    metadata_liq, prices_liq = _liquid_universe(metadata, prices_full)
    latest_date = prices_liq.index[-1]

    # Matching carta
    matches = find_best_card_match(card_query, metadata, prices_full)
    if not matches:
        print(f"❌ Nessuna carta trovata corrispondente a '{card_query}' nel catalogo.")
        return

    # Seleziona la prima (o più rilevante)
    item_id, info = matches[0]
    if len(matches) > 1:
        print(f"ℹ️ Trovate {len(matches)} corrispondenze. Valutazione su: '{info.get('name')}' ({info.get('game_slug')})")

    # Determina l'era
    if era_override:
        era = normalize_era(era_override)
    else:
        release_date = info.get("release_date", "2020-01-01")
        year = int(str(release_date)[:4])
        if year <= 2003:
            era = Era.VINTAGE
        elif year <= 2016:
            era = Era.MID_ERA
        else:
            era = Era.MODERN

    # Calcolo residuo e modello PokeQuant su data più recente
    strat = ScarcityValueFactorStrategy(**PRODUCTION_PARAMS)
    snap = _snapshot_for_date(prices_liq, metadata_liq, latest_date)
    residuals = strat._fit_residuals(pd.to_datetime(latest_date), snap)
    
    all_res = sorted(residuals.values())
    n_buy = max(1, int(len(residuals) * strat.top_quantile))
    ranked_top = sorted(residuals.items(), key=lambda x: x[1])[:n_buy][: strat.max_positions]
    cutoff_res = ranked_top[-1][1] if ranked_top else -1.1550

    if item_id in snap:
        base_psa_price = snap[item_id]["current_price"]
        res = residuals.get(item_id, -1.0)
        rank = sorted(residuals.items(), key=lambda x: x[1]).index((item_id, res)) + 1
        pct = rank / len(all_res) * 100.0
        # Prezzo max edge modello standard PSA 9
        theo_cutoff_price = base_psa_price * np.exp(cutoff_res - res)
        base_max_edge_price = base_psa_price + PRESERVE_EDGE_ALPHA * (theo_cutoff_price - base_psa_price)
    else:
        # Fallback se carta fuori dal panel liquido
        base_psa_price = float(prices_full[item_id].dropna().iloc[-1]) if item_id in prices_full else 100.0
        base_max_edge_price = base_psa_price * 1.05
        res = -1.0
        pct = 10.0

    # Se variante speciale, interroga prima PriceCharting per il prezzo Grade 9 reale
    pc_data = None
    v_key = variant_to_pricecharting_key(variant)
    if v_key:
        pc_data = fetch_pricecharting_variant_grade9(info.get("game_slug", ""), info.get("item_slug", ""), v_key)

    if pc_data:
        pc_eur, pc_usd, pc_url = pc_data
        base_psa_price = pc_eur
        base_max_edge_price = round(pc_eur * 1.05, 2)
        v_desc = f"PriceCharting Grado 9 reale (${pc_usd:.2f} USD)"
        orig_px = snap[item_id]["current_price"] if item_id in snap else base_psa_price
        v_mult = round(pc_eur / orig_px, 2) if orig_px > 0 else 1.0
    else:
        v_mult, v_desc = get_variant_multiplier(variant, info.get("game_slug"))
        base_psa_price = round(base_psa_price * v_mult, 2)
        base_max_edge_price = round(base_max_edge_price * v_mult, 2)

    # Ricalibrazione per la compagnia e grado scelti
    fair_value, sniper_ceiling_raw, adj = adjust_price_for_grading(
        base_psa_price_eur=base_psa_price,
        company=company,
        grade=grade,
        era=era,
        subgrades_black_label=is_black_label,
        is_pristine=is_pristine,
    )

    # Scala il tetto massimo del modello col coefficiente dello sniper
    calibrated_max_edge_allin = round(base_max_edge_price * adj.sniper_ceiling_factor, 2)
    
    if is_usa:
        landed_offer = estimate_usa_import_landed_cost(price_eur, item_type="single")
        allin_offer = landed_offer
        # Calcolo max offerta netta all'asta USA per non sforare il tetto sdoganato
        fixed_customs = (IMPORT_FROM_USA.intl_shipping_single_eur * (1.0 + IMPORT_FROM_USA.vat_rate)) + IMPORT_FROM_USA.eu_customs_duty_flat_eur + IMPORT_FROM_USA.courier_handling_fee_eur
        max_bid_usa = max(0.0, round((calibrated_max_edge_allin - fixed_customs) / (1.0 + IMPORT_FROM_USA.vat_rate), 2))
        sniper_net_ceiling = max_bid_usa
    else:
        allin_offer = price_eur + shipping_eur
        sniper_net_ceiling = max(0.0, round(calibrated_max_edge_allin - shipping_eur, 2))

    discount_pct = (1.0 - (allin_offer / fair_value)) * 100.0

    # Verdetto
    if allin_offer <= fair_value * 0.75:
        verdict = "🚨 DEEP VALUE / COLPACCIO (-25%+ di sconto)"
        verdict_color = "🟢"
    elif allin_offer <= fair_value * 0.95:
        verdict = "🟢 BUY CONSIGLIATO (A Sconto)"
        verdict_color = "🟢"
    elif allin_offer <= calibrated_max_edge_allin:
        verdict = "🟡 FAIR VALUE / AL LIMITE DELL'EDGE"
        verdict_color = "🟡"
    else:
        over_pct = ((allin_offer / calibrated_max_edge_allin) - 1.0) * 100.0
        verdict = f"🔴 SCARTARE / OVERPRICED (+{over_pct:.1f}% sopra il tetto)"
        verdict_color = "🔴"

    # Stampa del report
    print("\n" + "=" * 76)
    print(f"   VALUTAZIONE QUANTITATIVA SLAB — POKEQUANT VALUATION DESK")
    print("=" * 76)
    print(f"• Carta:            {info.get('name')} [{info.get('game_slug')}]")
    if v_mult > 1.0:
        print(f"• Variante/Edizione:{v_desc} (Moltiplicatore: {v_mult:.2f}x)")
    print(f"• Era Collez.:      {era.value.upper()} (Rilascio: {info.get('release_date', 'N/A')})")
    print(f"• Slab in Esame:    {adj.company.value} {grade} ({'Black Label Quad 10' if is_black_label else ('Pristine 10' if is_pristine else 'Standard')})")
    if is_usa:
        print(f"• Offerta Attuale:  {price_eur:.2f} € -> {allin_offer:.2f} € All-in Sdoganato da USA (IVA 22% + dazi)")
    else:
        print(f"• Offerta Attuale:  {price_eur:.2f} € (+ {shipping_eur:.2f} € sped.) = {allin_offer:.2f} € All-in")
    print("-" * 76)
    print(f"• Benchmark PSA:    {base_psa_price:.2f} € ({adj.benchmark_ref})")
    print(f"• Moltiplicatore:   {adj.multiplier:.3f}x (Penalità liquidità: -{adj.liquidity_penalty_pct:.1f}%)")
    print(f"• Fair Value Slab:  {fair_value:.2f} € (Valore reale atteso)")
    print(f"• Tetto Max Edge:   {calibrated_max_edge_allin:.2f} € (Costo totale massimo ammissibile)")
    print("-" * 76)
    if is_usa:
        print(f"🎯 MAX PUNTATA CONSIGLIATA SU EBAY USA: {sniper_net_ceiling:.2f} € (per non sforare a dogana)")
    else:
        print(f"🎯 PREZZO MASSIMO DA METTERE NELLO SNIPER: {sniper_net_ceiling:.2f} € (netto spedizione)")
    print(f"📊 Sconto Reale:    {discount_pct:+.1f}% rispetto al Fair Value")
    print(f"⚖️ Verdetto:         {verdict_color} {verdict}")
    print(f"📝 Note Modello:    {adj.notes}")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PokeQuant Slab Valuation & Multiplier Desk")
    parser.add_argument("--card", required=True, help="Nome o ID della carta (es. 'kabutops', 'azumarill 114')")
    parser.add_argument("--company", required=True, help="Casa di gradazione (PSA, BGS, CGC, SGC, GRAAD, PCA, ACE, TAG, CCC, AiGrading)")
    parser.add_argument("--grade", required=True, help="Voto slab (es. 9, 9.5, 10)")
    parser.add_argument("--price", type=float, required=True, help="Prezzo carta proposto/attuale in EUR")
    parser.add_argument("--shipping", type=float, default=0.0, help="Spese di spedizione in EUR (default: 0.0)")
    parser.add_argument("--variant", default="standard", help="Variante speciale (standard, 1st_edition, 1st_edition_base, no_symbol, shadowless, reverse_holo)")
    parser.add_argument("--era", choices=["vintage", "mid_era", "modern"], default=None, help="Override era collezionistica")
    parser.add_argument("--pristine", action="store_true", help="Flag per grado Pristine 10")
    parser.add_argument("--black-label", action="store_true", help="Flag per BGS 10 Black Label Quad 10")
    parser.add_argument("--usa", action="store_true", help="Flag se inserzione proviene da USA / extra-UE (applica dazi, IVA dogana 22% e oneri corriere)")

    args = parser.parse_args()
    evaluate_listing(
        card_query=args.card,
        company=args.company,
        grade=args.grade,
        price_eur=args.price,
        shipping_eur=args.shipping,
        era_override=args.era,
        is_pristine=args.pristine,
        is_black_label=args.black_label,
        variant=args.variant,
        is_usa=args.usa,
    )


if __name__ == "__main__":
    main()
