"""
poke_quant/config.py — Parametri globali di mercato e frizioni reali.
"""

from dataclasses import dataclass
from typing import Dict

# Tassi di cambio medi EUR/USD per conversione coerente
DEFAULT_EUR_USD = 1.08

# Commissioni percentuali e fisse delle piattaforme di vendita
PLATFORM_FEES: Dict[str, Dict[str, float]] = {
    "cardmarket": {
        "percentage": 0.05,        # 5% commissione base venditore Cardmarket
        "fixed_fee": 0.0,
        "payment_fee_pct": 0.0,    # a carico acquirente
    },
    "ebay": {
        "percentage": 0.125,       # 12.5% medio categoria collezionabili/carte
        "fixed_fee": 0.35,         # 0.35 EUR fisso per ordine
        "payment_fee_pct": 0.0,
    },
    "direct_private": {
        "percentage": 0.0,         # Scambio a mano / fiere / gruppi senza intermediari
        "fixed_fee": 0.0,
        "payment_fee_pct": 0.0,
    }
}

# Costi di spedizione e imballaggio tracciato/assicurato (Italia / EU)
SHIPPING_COSTS: Dict[str, float] = {
    "single_tracked": 7.00,       # Raccomandata / Corriere con tracking (toploader + cardboard)
    "sealed_box": 10.00,          # Pacco standard assicurato (bubble wrap + scatola rigida)
    "packaging_material": 0.60,   # Busta imbottita, toploader, sleeve, scatola
}

# Parametri di grading PSA (costo unitario comprensivo di spedizione round-trip e dogana)
# AGGIORNATO (2026-09-25, ricerca web su richiesta "considerando tutti i costi, ha senso
# sfruttare lo spread tra case di gradazione? Fai dei test" - vedi
# scripts/grading_company_crossover_arbitrage_test.py): i 25€/2 mesi precedenti erano
# STALE. PSA ha sospeso i tier economici (Value $29-59) il 2 giugno 2026 per arretrato -
# oggi il tier più economico disponibile e' Standard $59.99, turnaround 90-100 giorni
# lavorativi (~4.5-5 mesi), non piu' i 2 mesi assunti. Fonti: allvintagecards.com/
# psa-grading-costs/, cardgrade.io/psa-grading. Nuovo fee_per_card_eur: (59.99 fee +
# 10 handling + 15 return-ship-insured)$ / 1.08 EURUSD + 15€ spedizione IT->hub
# assicurata = ~94€, arrotondato a 95€. GradingArbitrageStrategy resta comunque SOLO
# SEGNALE ESPLORATIVO (HUMAN_RISK_TIER = "HIGH", mai eseguita in backtest/produzione) -
# questo aggiornamento corregge solo l'input costo, non valida la strategia.
@dataclass(frozen=True)
class GradingConfig:
    fee_per_card_eur: float = 95.00     # Costo totale all-in (fee PSA Standard + handling + spedizioni IT<->USA/EU)
    turnaround_months: int = 5          # Mesi di fermo del capitale prima che la carta sia vendibile (90-100gg lavorativi)
    default_modern_gem_rate: float = 0.70 # Probabilità stima PSA 10 su carte modern pack-fresh
    default_vintage_gem_rate: float = 0.20 # Probabilità stima PSA 10 su carte vintage (WotC)

GRADING_DEFAULT = GradingConfig()

# Costo di importazione da venditore extra-UE (USA - TCGplayer/eBay.com), richiesto
# esplicitamente dall'utente: "senza poter valutare gradazioni/lingue diverse ho
# poche opportunità dall'Italia, bisogna limare per capire il prezzo finale del
# mio mercato". Normativa verificata al 2026-09-24 (Regolamento UE 382/2026):
#   - Dal 1° luglio 2026 abolita la franchigia doganale a 150€ - OGNI spedizione
#     extra-UE paga dazio, indipendentemente dal valore.
#   - Dazio forfettario UE transitorio: 3€ per voce merceologica (fino al 2028).
#   - IVA all'importazione 22% (Italia) su valore + spedizione - dovuta su
#     qualsiasi importo dal 2021, nessuna soglia.
#   - Contributo nazionale italiano aggiuntivo 2€ per spedizioni <=150€ - DAL
#     1 OTTOBRE 2026, non ancora attivo alla data di questa nota.
#   - Commissione di sdoganamento del corriere (DHL/UPS/FedEx/Poste): flat,
#     NON normata, varia molto per corriere - stima indicativa, non un dato
#     ufficiale come le voci sopra.
# Confidenza: alta su IVA/dazio/franchigia (fonti multiple concordanti), bassa
# sulla commissione corriere (stima) - verificare sempre col corriere reale
# prima di trattare questo numero come vincolante.
@dataclass(frozen=True)
class ImportFromUsaConfig:
    vat_rate: float = 0.22
    eu_customs_duty_flat_eur: float = 3.00
    italy_national_contribution_eur: float = 2.00   # dal 2026-10-01, non ancora attivo oggi
    courier_handling_fee_eur: float = 15.00          # stima indicativa, non normata
    intl_shipping_single_eur: float = 20.00          # tracciato/assicurato, indicativo
    intl_shipping_sealed_eur: float = 35.00          # tracciato/assicurato, indicativo

IMPORT_FROM_USA = ImportFromUsaConfig()


def estimate_usa_import_landed_cost(
    item_price_eur: float, item_type: str = "single",
    include_national_contribution: bool = False,
) -> float:
    """Costo sdoganato stimato per comprare da un venditore USA (TCGplayer/
    eBay.com) e farsi spedire in Italia - oggetto + spedizione internazionale
    + IVA 22% (su oggetto+spedizione) + dazio forfettario UE 3€ + commissione
    di sdoganamento del corriere. include_national_contribution=False di
    default: il contributo italiano da 2€ entra in vigore il 2026-10-01, non
    ancora attivo alla data di scrittura - passare True dopo quella data.
    Stima, non un preventivo - la commissione corriere in particolare varia."""
    cfg = IMPORT_FROM_USA
    intl_shipping = cfg.intl_shipping_sealed_eur if item_type == "sealed" else cfg.intl_shipping_single_eur
    taxable_base = item_price_eur + intl_shipping
    vat = taxable_base * cfg.vat_rate
    total = taxable_base + vat + cfg.eu_customs_duty_flat_eur + cfg.courier_handling_fee_eur
    if include_national_contribution:
        total += cfg.italy_national_contribution_eur
    return round(total, 2)
