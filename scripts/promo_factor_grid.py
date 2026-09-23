"""
scripts/promo_factor_grid.py — Test del fattore "is_promo" (carte SVP/Black Star Promos)
con la stessa griglia + PBO + DSR + bootstrap + walk-forward H1/H2 usata su ogni altro
candidato di questa ricerca. Vedi l'esito completo nel docstring di
poke_quant/engine/strategies/rarity_tier_factor.py: numeri full-sample migliori di
qualsiasi altro fattore testato (Sharpe 1.27, PBO 0.000, DSR 0.980), ma NON VALIDATO
perché il walk-forward mostra un'inversione di segno netta (H1 Sharpe -0.95 -> H2 +1.93)
- lo stesso super-ciclo boom/bust ereditato da ogni fattore long-biased su questo
universo, qui solo amplificato dalla minore liquidità del sottomercato promo.
"""
import sys
import numpy as np
sys.path.insert(0, '.')
from poke_quant.data.storage import load_metadata, load_price_matrix
from poke_quant.engine.strategies.rarity_tier_factor import RarityTierFactorStrategy
from poke_quant.validation.statistical_validation import deflated_sharpe_ratio, pbo_cscv
from poke_quant.validation.bootstrap import block_bootstrap_metrics, summarize_bootstrap
from scripts.optimize_and_falsify import run_bt

metadata = load_metadata()
prices_full = load_price_matrix('historical_prices_graded_singles_grade9.csv')
singles_ids = [k for k, v in metadata.items() if v.get('type') == 'single' and v.get('data_quality') != 'thin_unreliable' and k in prices_full.columns]
meta_sub = {k: v for k, v in metadata.items() if k in singles_ids}
prices_sub = prices_full[singles_ids]
for k in meta_sub:
    meta_sub[k] = dict(meta_sub[k])
    meta_sub[k]['is_promo_str'] = 'yes' if meta_sub[k].get('is_promo') else 'no'
print(f"Universo: {len(singles_ids)} singole, promo tra queste: {sum(1 for v in meta_sub.values() if v.get('is_promo'))}", flush=True)

configs = {
    "PROMO rebal=6 minage=6": dict(rebalance_every_months=6, min_age_months=6),
    "PROMO rebal=3 minage=6": dict(rebalance_every_months=3, min_age_months=6),
    "PROMO rebal=6 minage=12": dict(rebalance_every_months=12, min_age_months=6),
}
results = {}
for name, kw in configs.items():
    strat = RarityTierFactorStrategy(field_name="is_promo_str", premium_rarities=frozenset({"yes"}), max_positions=999, **kw)
    res = run_bt(strat, prices_sub, meta_sub)
    results[name] = res
    print(f"  {name:26s} | CAGR {res.cagr*100:+6.2f}% | Sharpe {res.sharpe:5.2f} | MaxDD {res.max_drawdown*100:6.2f}% | Trade {res.total_trades:4d}", flush=True)

common_idx = None
for res in results.values():
    common_idx = res.monthly_returns.index if common_idx is None else common_idx.intersection(res.monthly_returns.index)
perf_matrix = np.column_stack([results[k].monthly_returns.loc[common_idx].values for k in configs])
t_len = len(perf_matrix)
splits = 8 if t_len >= 32 else 4
rem = t_len % splits
pbo = pbo_cscv(perf_matrix[rem:, :] if rem else perf_matrix, n_splits=splits)
print(f"\nPBO ({len(configs)} candidati, {splits} split): {pbo:.3f}", flush=True)
best_name = max(results, key=lambda k: results[k].sharpe)
best = results[best_name]
dsr = deflated_sharpe_ratio(observed_sr=best.sharpe/np.sqrt(12), n_trials=len(configs), n_obs=len(best.monthly_returns))
print(f"Migliore: '{best_name}' Sharpe {best.sharpe:.2f} | DSR: {dsr:.3f}", flush=True)
sims = block_bootstrap_metrics(best.monthly_returns, n_sims=500, block_size=6)
print(summarize_bootstrap(sims), flush=True)

mid = len(prices_sub) // 2
h1_dates, h2_dates = prices_sub.index[:mid], prices_sub.index[mid:]
print(f"\nWalk-forward H1/H2 sul vincitore '{best_name}':", flush=True)
for label, dates in [("H1", h1_dates), ("H2", h2_dates)]:
    sub = prices_sub.loc[dates]
    strat = RarityTierFactorStrategy(field_name="is_promo_str", premium_rarities=frozenset({"yes"}), max_positions=999, **configs[best_name])
    res = run_bt(strat, sub, meta_sub)
    print(f"  {label} ({dates[0].strftime('%Y-%m')}->{dates[-1].strftime('%Y-%m')}) | Sharpe {res.sharpe:5.2f} | CAGR {res.cagr*100:+6.2f}%", flush=True)
