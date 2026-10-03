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
    PSA_BLEND_PREMIUM_FACTOR,
    get_variant_multiplier,
    variant_to_pricecharting_key,
    SPECIAL_VARIANTS,
    ERA_PSA10_TO_PSA9_RATIO,
    ERA_BGS95_TO_PSA9_RATIO,
    estimate_psa10_from_psa9,
    estimate_grade95_from_psa9,
    get_recommended_grade_for_card,
    load_grade_ladder_cache,
)
from poke_quant.data.liquidity_filter import check_grade_ladder_tier_reliable
from poke_quant.config import estimate_usa_import_landed_cost, IMPORT_FROM_USA
from poke_quant.data.price_fetcher import (
    fetch_pricecharting_variant_grade9,
    fetch_pricecharting_variant_grade_tier,
    fetch_pricecharting_grade_tier_price,
)
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

    if item_id in snap and item_id in residuals:
        base_psa_price = snap[item_id]["current_price"]
        res = residuals[item_id]
        sorted_res = sorted(residuals.items(), key=lambda x: x[1])
        rank = sorted_res.index((item_id, res)) + 1
        pct = rank / len(all_res) * 100.0
        # Prezzo max edge modello standard PSA 9
        theo_cutoff_price = base_psa_price * np.exp(cutoff_res - res)
        base_max_edge_price = base_psa_price + PRESERVE_EDGE_ALPHA * (theo_cutoff_price - base_psa_price)
    else:
        # Fallback se carta fuori dal panel liquido o non nei residui
        base_psa_price = float(snap[item_id]["current_price"]) if item_id in snap else (float(prices_full[item_id].dropna().iloc[-1]) if item_id in prices_full else 100.0)
        base_max_edge_price = base_psa_price * 1.05
        res = -1.0
        pct = 10.0

    # GAP NOTO CHIUSO (dichiarato esplicitamente come non risolto nel fix
    # PSA_BLEND_PREMIUM_FACTOR, 2026-10-02 - l'utente: "cambiando il fair
    # value, bisogna testare anche il prezzo massimo per edge", poi: "tackle
    # into known, deliberately unfixed gap", 2026-10-03): base_psa_price qui
    # sopra (snap[item_id]["current_price"]) e' lo stesso identico pannello
    # PriceCharting Grado 9 BLEND cross-company usato per il resto del
    # modello - quando nessun dato reale per-grado/variante viene trovato
    # piu' sotto (is_grade_benchmark_resolved resta False), questo valore
    # finiva nel ramo "algoritmico" di adjust_price_for_grading() SENZA mai
    # ricevere il premio PSA_BLEND_PREMIUM_FACTOR (quel ramo assume che il
    # prezzo passato sia GIA' il vero prezzo PSA, non un blend) - lo stesso
    # bias quindi si propagava silenziosamente a OGNI stima algoritmica
    # senza dato reale, non solo al caso "PSA grado 9.0 esatto" citato nel
    # fix originale (EMPIRICAL_RATIOS_GRADE9[(PSA,"9.0",era)]=1.0 era solo il
    # sintomo piu' visibile: lo stesso blend non corretto alimenta OGNI
    # company/grado di quella tabella, che e' calibrata assumendo "base
    # PSA9" = vero prezzo PSA, non blend).
    #
    # Applicato QUI, DOPO il calcolo di scarsita' sopra (res/pct/cutoff_res
    # sono gia' calcolati e NON vengono toccati) - DELIBERATAMENTE non alla
    # lettura di snap[item_id]["current_price"] piu' in alto: quel valore
    # alimenta anche _fit_residuals() sull'intero pannello liquido, dove
    # un bias asimmetrico su una sola carta (diverso da tutte le altre nello
    # stesso identico calcolo di rank relativo) romperebbe la semantica del
    # modello di scarsita' - diverso dal caso gia' verificato in
    # scripts/eu_price_level_invariance_test.py (un bias UNIFORME su TUTTO
    # il pannello non altera il ranking). Sovrascritto piu' sotto senza
    # effetto se un dato reale per-grado/variante viene trovato (pc_eur
    # fresco rimpiazza interamente questo valore, nessun doppio conteggio).
    base_psa_price = round(base_psa_price * PSA_BLEND_PREMIUM_FACTOR, 2)
    base_max_edge_price = round(base_max_edge_price * PSA_BLEND_PREMIUM_FACTOR, 2)

    # Risoluzione benchmark per il grado scelto (da 10.0 fino a 7.0) - PRIMA
    # della risoluzione della variante, perche' ora serve per interrogare il
    # dato reale della variante ESATTAMENTE al grado richiesto (vedi sotto).
    grade_str = str(grade).strip().lower().replace("_", ".")
    is_grade_benchmark_resolved = False

    if "10" in grade_str:
        grade_tier = "psa10"
        tier_label = "PSA 10"
    elif "9.5" in grade_str or "95" in grade_str:
        grade_tier = "grade9_5"
        tier_label = "Grado 9.5"
    elif "8.5" in grade_str or "85" in grade_str:
        grade_tier = "grade8_5"
        tier_label = "Grado 8.5"
    elif "8" in grade_str:
        grade_tier = "grade8"
        tier_label = "Grado 8.0"
    elif "7.5" in grade_str or "75" in grade_str:
        grade_tier = "grade7_5"
        tier_label = "Grado 7.5"
    elif "7" in grade_str:
        grade_tier = "grade7"
        tier_label = "Grado 7.0"
    else:
        grade_tier = "grade9"
        tier_label = "Grado 9.0"

    # BUCO STRUTTURALE TROVATO (l'utente: "tappa il buco del filtro mercato
    # sottile", 2026-09-30): compute_thin_market_drift_flags() copre solo il
    # pannello legacy grade9(PSA9)/raw - un dato reale per un grado specifico
    # (9.5/10/8/7, da data_cache/grade_ladder_prices.json) non passava da
    # NESSUN controllo di attendibilita'. Verifica qui la serie storica per
    # QUESTO grado specifico, indipendentemente da come il benchmark sotto
    # viene infine risolto - vedi poke_quant.data.liquidity_filter::
    # check_grade_ladder_tier_reliable per la calibrazione.
    ladder_flag_reason = None
    ladder_series = load_grade_ladder_cache().get(item_id, {}).get(grade_tier)
    if ladder_series:
        _ladder_ok, _ladder_reason = check_grade_ladder_tier_reliable(ladder_series, tier=grade_tier)
        if not _ladder_ok:
            ladder_flag_reason = _ladder_reason

    # Se variante speciale, prova PRIMA il dato reale PriceCharting per il
    # grado ESATTO richiesto sulla pagina della variante (richiesta esplicita
    # dell'utente dopo il bug del voto slab non aggiornante il prezzo,
    # 2026-09-29: "deve prendere i dati reali quanto possibile") - solo se
    # quella pagina non ha un dato per QUESTO grado specifico si cade sul
    # Grado 9 reale (sempre disponibile se la variante ha una pagina) + la
    # stessa scalatura del ramo algoritmico, come gia' corretto in precedenza.
    pc_data = None
    pc_variant_tier = None
    v_key = variant_to_pricecharting_key(variant)
    if v_key:
        pc_variant_tier = fetch_pricecharting_variant_grade_tier(
            info.get("game_slug", ""), info.get("item_slug", ""), v_key, tier=grade_tier
        )
        if not pc_variant_tier:
            pc_data = fetch_pricecharting_variant_grade9(info.get("game_slug", ""), info.get("item_slug", ""), v_key)

    if pc_variant_tier:
        pc_eur, pc_usd, pc_url = pc_variant_tier
        base_psa_price = pc_eur
        base_max_edge_price = round(pc_eur * 1.05, 2)
        v_desc = f"PriceCharting Variante Reale {tier_label} (${pc_usd:.2f} USD)"
        benchmark_note = v_desc
        v_mult = 1.0  # il prezzo e' gia' quello esatto della variante a questo grado
        is_grade_benchmark_resolved = True
    elif pc_data:
        pc_eur, pc_usd, pc_url = pc_data
        base_psa_price = pc_eur
        base_max_edge_price = round(pc_eur * 1.05, 2)
        v_desc = f"PriceCharting Grado 9 reale (${pc_usd:.2f} USD)"
        orig_px = snap[item_id]["current_price"] if item_id in snap else base_psa_price
        v_mult = round(pc_eur / orig_px, 2) if orig_px > 0 else 1.0
        benchmark_note = f"PriceCharting Grado 9 reale (${pc_usd:.2f} USD)"
    else:
        v_mult, v_desc = get_variant_multiplier(variant, info.get("game_slug"))
        base_psa_price = round(base_psa_price * v_mult, 2)
        base_max_edge_price = round(base_max_edge_price * v_mult, 2)
        benchmark_note = f"Prezzo mercato PSA 9 (blend PriceCharting × {PSA_BLEND_PREMIUM_FACTOR:.2f} premio PSA)"

    # BUG TROVATO (richiesta esplicita dell'utente: valutazione di acquisti reali,
    # "e' importante mantenere edge", 2026-09-29): quando una variante speciale
    # (1st Edition/No Symbol/Shadowless) aveva gia' un prezzo REALE trovato sopra
    # (pc_variant_tier o pc_data, dalla pagina dedicata della variante), questo
    # fetch successivo interrogava SEMPRE anche la pagina della stampa STANDARD
    # (game_slug/item_slug, senza variante) e - se trovava un prezzo - lo usava
    # per SOVRASCRIVERE silenziosamente base_psa_price, buttando via il prezzo
    # reale della variante gia' trovato. Risultato concreto: una CGC 9.0 1st
    # Edition Gym Challenge Sabrina #20 (prezzo reale 1st Ed. $229.99) veniva
    # valutata contro il benchmark della stampa Unlimited ($106.12), producendo
    # un falso "SCARTARE / OVERPRICED +72.6%" su una carta che al benchmark
    # corretto era vicina al fair value. Stessa guardia gia' presente in app.py
    # (is_pc_grade_resolved) - qui mancava. Se un prezzo di variante era gia'
    # risolto (a qualunque grado), il fetch generico va saltato.
    pc_tier = None if (pc_variant_tier or pc_data) else fetch_pricecharting_grade_tier_price(
        info.get("game_slug", ""), info.get("item_slug", ""), tier=grade_tier, item_id=item_id
    )
    if pc_tier:
        pc_eur, pc_usd, pc_url, pc_source = pc_tier
        # BUG TROVATO (indagando "Jolteon PSA 7 no symbol mi dice Fair value
        # PSA 151.01 EUR, ma su pricecharting e' molto piu' basso",
        # 2026-09-30): quando NESSUNA pagina PriceCharting dedicata esiste
        # per la variante scelta (pc_variant_tier e pc_data entrambi None -
        # tipico se la variante non esiste per quella specifica ristampa),
        # questo ramo prendeva comunque il prezzo reale della stampa
        # STANDARD e marcava is_grade_benchmark_resolved=True senza mai
        # applicare la stima del premio di variante (v_mult restava quello
        # di default 1.0 dal ramo "else" sopra, gia' eseguito PRIMA che
        # questo fetch lo sovrascrivesse) - il premio (es. ~1.4x No Symbol)
        # spariva silenziosamente, pur etichettando il risultato come un
        # dato reale valido per quella variante. Corretto applicando qui la
        # stima del moltiplicatore quando la variante e' speciale ma non
        # abbinata a un dato reale specifico.
        if v_key and not pc_variant_tier and not pc_data:
            v_mult, v_desc = get_variant_multiplier(variant, info.get("game_slug"))
            base_psa_price = round(pc_eur * v_mult, 2)
            base_max_edge_price = round(pc_eur * v_mult * 1.05, 2)
            benchmark_note = f"Dato Reale {pc_source} Stampa Standard ({tier_label}: ${pc_usd:.2f} USD) × stima variante {v_mult:.2f}x"
        else:
            base_psa_price = pc_eur
            base_max_edge_price = round(pc_eur * 1.05, 2)
            benchmark_note = f"Dato Reale {pc_source} ({tier_label}: ${pc_usd:.2f} USD)"
        is_grade_benchmark_resolved = True
    elif pc_data:
        # BUG TROVATO (l'utente: "modificare il voto slab non modifica i prezzi
        # consigliati", 2026-09-29): fetch_pricecharting_variant_grade9()
        # restituisce SEMPRE il prezzo reale al grado 9.0 della variante
        # (legge la chiave "graded" del JSON PriceCharting, indipendente dal
        # voto scelto) - ma veniva trattato come benchmark GIA' risolto per
        # QUALUNQUE voto (is_grade_benchmark_resolved=True incondizionato),
        # saltando la scalatura di grado sotto e passando
        # is_grade_benchmark_price=True anche per un 10.0: il prezzo restava
        # bloccato al valore Grado 9 ma ETICHETTATO come "PSA_10", producendo
        # per la STESSA carta un fair value PSA10 piu' BASSO del fair value
        # PSA9 (verificato: Jolteon #4 No Symbol Error, CGC 9.0 -> 283,46€,
        # CGC 10.0 -> 134,65€, invertito). Risolto SOLO se il voto scelto e'
        # davvero 9.0 (l'unico che il fetch restituisce); per ogni altro voto
        # si riusa questo prezzo reale come base PSA9 (un dato reale, non un
        # fallback peggiore) e si applica sotto la stessa scalatura del ramo
        # puramente algoritmico - senza rientrare nel fetch generico sopra
        # (gia' saltato per costruzione quando pc_data esiste), che
        # sovrascriverebbe il prezzo della variante con quello della stampa
        # standard (bug distinto, gia' corretto in precedenza).
        is_grade_benchmark_resolved = "9" in grade_str and "9.5" not in grade_str and "95" not in grade_str
        if not is_grade_benchmark_resolved:
            benchmark_note = f"PriceCharting Variante Reale, base Grado 9 ({base_psa_price:.2f} €)"

    if not is_grade_benchmark_resolved and "10" in grade_str:
        p10_ratio = ERA_PSA10_TO_PSA9_RATIO.get(era, 3.00)
        base_psa_price = round(base_psa_price * p10_ratio, 2)
        base_max_edge_price = round(base_max_edge_price * p10_ratio, 2)
        benchmark_note = f"Stima algoritmica Grado 10 (Base PSA 9 × {p10_ratio:.2f}x)"
    elif not is_grade_benchmark_resolved and not pc_data:
        # BUG TROVATO (trovato indagando il bug sopra, stessa richiesta
        # dell'utente): questo ramo applicava una pre-scalatura
        # (ERA_BGS95_TO_PSA9_RATIO) per il grado 9.5 e POI
        # adjust_price_for_grading applicava DI NUOVO un moltiplicatore gia'
        # "vs PSA9" da EMPIRICAL_RATIOS_GRADE9 (get_grading_adjustment
        # assegna benchmark_ref="PSA_9" anche al grado 9.5, non "PSA_10" -
        # verificato leggendo il codice) - doppio conteggio che sovrastimava
        # il fair value di un fattore pari esattamente a
        # ERA_BGS95_TO_PSA9_RATIO (es. CGC 9.5 Moderno: 114,72€ corretto vs
        # 184,69€ col doppio conteggio, +61%). Il grado 10 ne ha davvero
        # bisogno (EMPIRICAL_RATIOS_GRADE10 e' "vs PSA10", non "vs PSA9" -
        # PSA/10.0=1.0x esatto, "Benchmark base Grado 10"), ma 9.5/8.5/8.0/
        # 7.5/7.0 no: EMPIRICAL_RATIOS_GRADE9 e' gia' l'intero rapporto vs
        # PSA9, nessuna pre-scalatura va applicata qui (stesso trattamento
        # gia' corretto che il grado 8.5/8.0/7.5/7.0 riceveva).
        benchmark_note = f"Benchmark PokeQuant Base PSA 9 ({base_psa_price:.2f} €)"

    # Ricalibrazione per la compagnia e grado scelti
    fair_value, sniper_ceiling_raw, adj = adjust_price_for_grading(
        base_psa_price_eur=base_psa_price,
        company=company,
        grade=grade,
        era=era,
        subgrades_black_label=is_black_label,
        is_pristine=is_pristine,
        is_grade_benchmark_price=is_grade_benchmark_resolved,
    )

    # BUG TROVATO verificando il fix PSA_BLEND_PREMIUM_FACTOR (l'utente, 2026-10-02):
    # questa riga sovrascriveva SEMPRE calibrated_max_edge_allin con una
    # formula parallela (base_max_edge_price * adj.sniper_ceiling_factor),
    # ignorando sniper_ceiling_raw gia' calcolato correttamente sopra da
    # adjust_price_for_grading() - comprese correzioni come PSA_BLEND_
    # PREMIUM_FACTOR, applicate SOLO dentro quella funzione. Risultato
    # verificato: Fair Value si aggiornava (92,96€ per Dragonite-EX #106
    # PSA 9), ma "Tetto Max Edge" e il verdetto finale no (restavano a
    # 90,37€, il valore pre-fix) - la stessa carta mostrava un fair value
    # corretto ma una soglia di acquisto/verdetto ancora sbagliati. app.py
    # (la dashboard) non aveva questo problema: sovrascrive
    # sniper_ceiling_calib con la stessa formula parallela SOLO quando
    # `not is_price_grade_matched` (nessun dato reale per il grado esatto,
    # serve una stima algoritmica) - qui invece scattava sempre,
    # incondizionatamente. Allineato allo stesso schema di app.py.
    if is_grade_benchmark_resolved:
        calibrated_max_edge_allin = round(sniper_ceiling_raw, 2)
    else:
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
    # BUG TROVATO IN AUDIT (2026-09-29): is_grade_10/is_grade_95 non erano mai
    # definiti in questa funzione -> NameError certo ogni volta che si valuta
    # una variante speciale (1st edition/no-symbol/shadowless, v_mult>1.0).
    # Derivati qui da adj.benchmark_ref/grade, stesso parsing di
    # get_grading_adjustment() in grading_multipliers.py.
    is_grade_10 = adj.benchmark_ref == "PSA_10"
    is_grade_95 = "9.5" in str(grade).replace("_", ".") or "95" in str(grade).replace("_", ".")
    if v_mult > 1.0 and not is_grade_10 and not is_grade_95:
        print(f"• Variante/Edizione:{v_desc} (Moltiplicatore: {v_mult:.2f}x)")
    rec_grade = get_recommended_grade_for_card(era=era)
    print(f"• Era Collez.:      {era.value.upper()} (Rilascio: {info.get('release_date', 'N/A')})")
    print(f"• Target Grado:     {rec_grade['target_badge']} — {rec_grade['short_advice']}")
    print(f"• Slab in Esame:    {adj.company.value} {grade} ({'Black Label Quad 10' if is_black_label else ('Pristine 10' if is_pristine else 'Standard')})")
    if is_usa:
        print(f"• Offerta Attuale:  {price_eur:.2f} € -> {allin_offer:.2f} € All-in Sdoganato da USA (IVA 22% + dazi)")
    else:
        print(f"• Offerta Attuale:  {price_eur:.2f} € (+ {shipping_eur:.2f} € sped.) = {allin_offer:.2f} € All-in")
    print("-" * 76)
    print(f"• Benchmark PSA:    {base_psa_price:.2f} € ({adj.benchmark_ref} — {benchmark_note})")
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
    if era == Era.MODERN and ("9.0" in grade_str or grade_str == "9"):
        print("-" * 76)
        print("⚠️  ATTENZIONE LIQUIDITÀ MODERNO: Sulle carte moderne (2017+) il mercato assorbe")
        print("   quasi esclusivamente copie PSA 10 o Raw. Le slab Grado 9 moderne hanno turnover")
        print("   lento e scarso premio rispetto al Raw. Si raccomanda di puntare a PSA 10 o BGS 9.5.")
    if info.get("data_quality") == "thin_unreliable":
        print("-" * 76)
        print("🚩 ATTENZIONE MERCATO SOTTILE: questa carta è flaggata come dato inaffidabile")
        print(f"   (scripts/flag_unreliable_assets.py): {info.get('data_quality_reason', '')}")
        print("   Il benchmark sopra può essere gonfiato da poche vendite reali al grado.")
        print("   Verificare a mano prima di procedere (es. comp reali recenti sulla stessa carta).")
    if ladder_flag_reason:
        print("-" * 76)
        print(f"🚩 ATTENZIONE MERCATO SOTTILE SU {tier_label.upper()}: {ladder_flag_reason}")
        print("   Verificare a mano un comp reale recente per questo grado specifico prima di procedere.")
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
