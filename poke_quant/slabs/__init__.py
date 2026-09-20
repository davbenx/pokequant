"""
poke_quant/slabs — Modulo strategico, quantitativo e di alert per Carte Gradate (PSA, BGS, CGC).
"""

from poke_quant.slabs.models import (
    GradingCompany,
    SlabGrade,
    EdgeType,
    SignalAction,
    Subgrades,
    SlabQuote,
    SlabSignal,
    RotationRecommendation,
    SlabHolding
)
from poke_quant.slabs.edge_calculator import (
    calc_cross_grading_spread,
    calc_pop_saturation_edge,
    calc_gem_scarcity_edge,
    calc_manufacturing_cost_floor,
    calc_geo_dislocation,
    calc_pop_dilution_wave_sell,
    calc_parabolic_exhaustion_sell,
    calc_spread_convergence_sell,
    calc_opportunity_cost_rotation,
    calc_era_cycle_rotation
)
from poke_quant.slabs.slab_universe import get_slab_universe, get_curated_grails, get_failed_controls
from poke_quant.slabs.slab_scanner import scan_slabs_market
from poke_quant.slabs.slab_backtester import SlabBacktester, SlabBacktestResult
from poke_quant.slabs.slab_falsification import run_popperian_falsification_suite
from poke_quant.slabs.slab_notifier import format_slabs_telegram_alert, print_slabs_cli_summary

__all__ = [
    "GradingCompany",
    "SlabGrade",
    "EdgeType",
    "SignalAction",
    "Subgrades",
    "SlabQuote",
    "SlabSignal",
    "RotationRecommendation",
    "SlabHolding",
    "calc_cross_grading_spread",
    "calc_pop_saturation_edge",
    "calc_gem_scarcity_edge",
    "calc_manufacturing_cost_floor",
    "calc_geo_dislocation",
    "calc_pop_dilution_wave_sell",
    "calc_parabolic_exhaustion_sell",
    "calc_spread_convergence_sell",
    "calc_opportunity_cost_rotation",
    "calc_era_cycle_rotation",
    "get_slab_universe",
    "get_curated_grails",
    "get_failed_controls",
    "scan_slabs_market",
    "SlabBacktester",
    "SlabBacktestResult",
    "run_popperian_falsification_suite",
    "format_slabs_telegram_alert",
    "print_slabs_cli_summary"
]
