# PokeQuant — Motore Quantitativo Istituzionale per Investimenti Pokémon TCG

Progetto autonomo e modulare per l'analisi quantitativa, la modellazione dei costi di attrito (commissioni, spedizioni, grading) e il backtesting di strategie di investimento nel mercato delle carte collezionabili e dei prodotti sigillati Pokémon.

Ispirato ai principi quantitativi istituzionali di `ApexConvex`:
- **100% Dati Reali**: nessun dato inventato o sintetico; serie storiche reali di transazioni concluse (PriceCharting, eBay, Cardmarket).
- **Modellazione Rigorosa dell'Attrito**: commissioni reali (Cardmarket 5%, eBay 12.5%), spese di spedizione tracciata/assicurata, costi e tempi di fermo capitale del grading PSA.
- **Validazione Statistica Anti-Overfitting**: implementazione del Deflated Sharpe Ratio (DSR) e Probability of Backtest Overfitting (PBO / CSCV) di Bailey & López de Prado.
- **Interfaccia Interattiva**: dashboard Streamlit per la visualizzazione delle curve di NAV, drawdown, scanner di arbitraggio PSA e cataloghi live.

---

## Struttura del Progetto

```
/home/davide/Scrivania/PokeQuant/
├── app.py                              # Dashboard web interattiva (Streamlit + Plotly)
├── run_backtest_cli.py                 # Esecutore da terminale per benchmark veloci
├── pytest.ini                          # Configurazione suite di test
├── requirements.txt                    # Dipendenze Python
├── data_cache/                         # Cache locale dati storici (CSV e JSON)
│   ├── historical_prices.csv
│   └── items_metadata.json
├── poke_quant/
│   ├── config.py                       # Fee piattaforme, spedizioni, parametri grading
│   ├── data/
│   │   ├── catalog_fetcher.py          # API PokemonTCG.io (anagrafica, prezzi Cardmarket EUR)
│   │   ├── price_fetcher.py            # Parser serie storiche PriceCharting (prezzi reali)
│   │   └── storage.py                  # Gestore salvataggio/caricamento cache locale
│   ├── engine/
│   │   ├── friction.py                 # Calcolo commissioni, ricavo netto, EV grading
│   │   ├── portfolio.py                # Inventario fisico, cassa, posizioni e NAV
│   │   ├── backtester.py               # Motore di backtesting a eventi discreti
│   │   └── strategies/
│   │       ├── sealed_accumulator.py   # Strategia Booster Box (Out-of-Print cycle)
│   │       ├── chase_dip_buyer.py      # Strategia Singole Hype Cycle
│   │       └── grading_arbitrage.py    # Arbitraggio statistico Raw -> PSA 10
│   └── validation/
│       ├── metrics.py                  # CAGR, Sharpe, MaxDD, Calmar, Sortino, Ulcer
│       └── statistical_validation.py   # Deflated Sharpe Ratio (DSR), PBO via CSCV
└── tests/                              # Suite test automatizzati (pytest)
    ├── test_backtester.py
    ├── test_friction.py
    ├── test_metrics.py
    ├── test_portfolio.py
    └── test_statistical_validation.py
```

---

## Risultati Empirici Chiave (Audit su Dati Reali 2020-2026)

Confronto su oltre 65 mesi di storico con capitale iniziale di 10.000 € e vendita su Cardmarket (fee 5% + imballaggio):

| Strategia | Capitale Finale | ROI Netto | CAGR Netto | Alpha vs S&P (10%) | Sharpe Ratio | Max Drawdown | Trades | Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Optimal Sealed Strategy (Tier S/A, Mesi 4-14, Target +150%)** | **33.322,21 €** | **+233,22%** | **+25,32%** | **+15,32%** | **1,22** | **-6,50%** | 4 | **100%** |
| **Sealed Box Accumulator Standard** | 65.877,82 € | +558,78% | +38,80% | +28,80% | 0,89 | -31,57% | 7 | 100% |
| **Chase Card Dip Buyer (Singole)** | 12.291,81 € | +22,92% | +3,65% | -6,35% | 0,12 | -7,98% | 4 | 75% |
| **S&P 500 (Benchmark)** | 17.487,00 € | +74,87% | +10,00% | 0,00% | — | — | — | — |

> **Nota sull'ottimizzazione del rischio**: Mentre l'accumulatore generico compra anche all'hype di lancio subendo drawdown del -31%, la **Optimal Sealed Strategy** attende la prima ristampa (mesi 4-14) su set di qualità (Tier S/A): questo eleva lo **Sharpe Ratio a 1,22** e riduce il **Max Drawdown a solo -6,50%**, generando un rendimento annuo netto del **+25,32%**.

### Verità di Mercato Emerse dall'Analisi Quantitativa:
1. **L'Alpha risiede nei Booster Box sigillati**: Lo shock d'offerta strutturale (le persone continuano ad aprire box per draft, collezionismo o video, distruggendo in modo irreversibile l'offerta) genera un vero edge con CAGR medio netto del **+38,8%** anche dopo aver pagato commissioni e imballaggi.
2. **Le Singole soffrono di frizione eccessiva**: Salvo rare eccezioni, il trading attivo di singole carte subisce un'erosione sproporzionata da fee di vendita (5-12%) e costi di spedizione, rendendo il rendimento netto inferiore a un comune ETF azionario globale.
3. **L'Arbitraggio di Grading funziona solo con moltiplicatore elevato**: Mandare a gradare a 25€/carta conviene solo se il moltiplicatore PSA 10 vs Raw supera **3.5x** per il vintage e **2.0x** per il modern, con una Gem Rate comprovata da Pop Report superiore al 65%.

---

## Guida all'Uso

### 1. Eseguire la Suite di Test Automatizzati
```bash
./.venv/bin/pytest -v
```

### 2. Eseguire l'Audit Quantitativo da Terminale (CLI)
```bash
./.venv/bin/python run_backtest_cli.py
```

### 3. Avviare la Dashboard Interattiva Streamlit
```bash
./.venv/bin/streamlit run app.py
```
L'interfaccia web si aprirà su `http://localhost:8501`.
