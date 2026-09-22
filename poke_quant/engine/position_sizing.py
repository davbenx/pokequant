"""
poke_quant/engine/position_sizing.py — Regola esplicita di dimensionamento per età,
in sostituzione della nota qualitativa nel runbook ("pesa meno le release recenti").

Motivazione empirica (non uno sfizio, e non un parametro scelto guardando il
rendimento — solo la constatazione già riprodotta 3 volte in questa sessione su
dataset indipendenti): la coorte di asset rilasciati negli ultimi ~12-18 mesi ha
mostrato drawdown molto più ampi di quella matura (Equal-Weight: -49% sulla coorte
2023+ vs -13/-21% sulla coorte più vecchia). Qui si traduce quella constatazione in
un moltiplicatore di size, non in un filtro rigido (che in Fase 2 si è visto
distruggere la reattività della strategia sui set nuovi — vedi TimeSeriesMomentumStrategy
min_age_months=36 che azzerava i trade sulla coorte nuova).

Rampa lineare: size piena (1.0) solo dopo MATURITY_MONTHS; prima, cresce
linearmente da FLOOR_MULTIPLIER a 1.0. Non è "il numero corretto" — è una scelta
di margine di sicurezza esplicita e conservativa, documentata così com'è.
"""

from __future__ import annotations

FLOOR_MULTIPLIER = 0.4
MATURITY_MONTHS = 18


def age_weight(age_months: float, floor: float = FLOOR_MULTIPLIER, maturity_months: int = MATURITY_MONTHS) -> float:
    """Moltiplicatore di dimensionamento posizione in [floor, 1.0] in funzione dell'età."""
    if age_months is None or age_months < 0:
        return floor
    if age_months >= maturity_months:
        return 1.0
    return floor + (1.0 - floor) * (age_months / maturity_months)
