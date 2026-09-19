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

## Risultati Empirici su Dataset Esteso (53 Asset Reali, 68 Mesi 2021-2026)

Confronto su 68 mesi di serie storiche reali di transazioni concluse con capitale iniziale di 10.000 €, commissioni Cardmarket (5%), costi di packaging, slippage di liquidità (1-2%) e costi vivi di custodia/storage (0.5%/anno):

| Strategia | Capitale Finale | ROI Netto | CAGR Netto | Alpha vs Real SPY | Sharpe Ratio | Max Drawdown | Beta vs SPY | Correlazione | Rotazioni Tranche 1 | Trades Chiusi | Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Optimal Sealed Strategy (Rotazione Dinamica Scalare)** | **50.269,12 €** | **+402,69%** | **+33,54%** | **+22,29%** | **1,68** | **-8,93%** | **0,13** | **0,12** | **15** | **23** | **100%** |
| **Sealed Box Accumulator Standard** | 26.332,62 € | +163,33% | +18,94% | +7,69% | 0,81 | -32,56% | -0.00 | -0.00 | 0 | 6 | 100% |
| **Chase Card Dip Buyer (Singole)** | 12.140,47 € | +21,40% | +3,53% | -7,71% | 0.11 | -8,06% | -0.01 | -0.01 | 0 | 4 | 75% |
| **S&P 500 ETF (SPY Benchmark Reale)** | 18.133,75 € | +81,34% | +11,25% | 0,00% | — | -24,50% | 1,00 | 1,00 | — | — | — |

### Meccanica della Rotazione Dinamica Scalare (Capital Rotation Engine):
1. **Tranche 1 (Rotazione Parziale / Sblocco Liquidità)**: a 18+ mesi dalla release (set confermato Out-of-Print) e al raggiungimento di un ROI Netto $\ge +70\%$, il sistema liquida il 50% delle unità. Questo recupera interamente il capitale investito più un profitto netto del 70%, liberando cassa per acquistare nuovi set moderni a MSRP durante la loro finestra di ristampa (Mesi 4-14).
2. **Tranche 2 (Moonbag / Ciclo Completo)**: il restante 50% delle unità viene mantenuto in detenzione fino al target $\ge +150\%$ (holding 30-44 mesi) o al time-stop di 48 mesi.
3. **Cap di Allocazione al 10-12%**: consente di detenere in contemporanea 8-10 posizioni attive (Tier S, A e B di qualità), eliminando il blocco della liquidità e garantendo un flusso continuo di segnali operativi (85 segnali generati distribuiti con continuità su 2021, 2022, 2023, 2024, 2025 e 2026).
4. **Inventario Attivo a Fine Periodo (Settembre 2026)**: 7 set in detenzione (*151 Bundle, Obsidian Flames, Shiny Treasure ex JP, Twilight Masquerade, Paradox Rift, Temporal Forces, OP-06 Wings of the Captain*) per un controvalore di mercato di **20.448 €** e una cassa libera di **29.821 €**.

### Rendimenti per Regime Macroeconomico:
- **BULL_HYP (2021 Post-Covid Bubble)**: -2,13% (Capitale preservato in attesa della correzione/ristampa)
- **BEAR_MACRO (2022 QT / Rialzo Tassi / Crypto Crash)**: **+22,54%** (contro il -18,1% dell'S&P 500)
- **ACCUMULATION (2023 Consolidamento)**: **+7,54%**
- **SELECTIVE_EXPANSION (2024-2026 Esaurimento Scorte & Rotazione)**: **+289,79%**

---

## Suite di Audit e Falsificazione Quantitativa (8 Test Popperiani)

1. **Anti-Outlier Test**: Escludendo i due massimi vincitori storici (*Evolving Skies* e *Team Up*), la strategia mantiene un **CAGR del +25,07%** (Alpha +15,07% vs benchmark passivo). L'edge non dipende da casi fortuiti isolati.
2. **Bear Market Test (Gen 2022 - 2024)**: Avviando il portafoglio all'inizio del Quantitative Tightening e del crollo degli asset alternativi, il **CAGR netto è del +29,42%** con un Max Drawdown contenuto a **-6,50%**.
3. **Friction Shock (eBay 12.5% + Spedizioni + 2% Slippage + Custodia)**: Anche con attriti elevati, il **CAGR netto si attesta al +21,70%** (+10,45% di Alpha netto su S&P 500 reale).
4. **Falsificazione Selezione Tier C**: Un portafoglio composto solo da set deboli (*Rebel Clash, Battle Styles, Crimson Invasion*) genera solo il **+5,38% CAGR** (-5,87% vs S&P 500), confermando il limite di demarcazione: l'out-of-print senza mascotte iconiche distrugge valore.
5. **Cross-TCG Generalization (One Piece TCG)**: Testando le stesse regole su Booster Box di One Piece (*OP-01 Romance Dawn, OP-02 Paramount War, OP-05 Awakening*), il rendimento netto è del **+17,99% CAGR** (+151,84% ROI), validando l'universalità della legge di distruzione irreversibile dell'offerta nei TCG da collezione.
6. **Resilienza Edizioni Giapponesi [JP]**: I set giapponesi High-Class (*VSTAR Universe, VMAX Climax, Tag All Stars*) evidenziano un CAGR ridotto (+3,01%) e un Max Drawdown maggiore (-28,08%) a causa del sovrapprezzo d'importazione iniziale in Europa e dei cicli di ristampa improvvisi in Giappone.
7. **Monte Carlo Noise Perturbation (100 Simulazioni a +-15% di rumore sui prezzi)**: Con il 95% di confidenza statistica, il CAGR peggiore nel 5° percentile è del **+21,51%** (ben superiore al benchmark azionario del 10%), provando la robustezza del modello contro la varianza dei prezzi di mercato.
8. **Covarianza e Beta Multi-Asset**: Correlazione di soli **+0,20 con S&P 500**, **+0,21 con Oro** e **+0,30 con Bitcoin**, con **Beta di 0,23**, a dimostrazione della natura di autentico asset alternativo decorrelato.

---

## Guida all'Uso

### 1. Eseguire la Suite di Test Automatizzati
```bash
./.venv/bin/pytest -v
```

### 2. Eseguire la Suite di 8 Test di Falsificazione
```bash
./.venv/bin/python poke_quant/falsification_suite.py
```

### 3. Eseguire l'Audit Quantitativo da Terminale (CLI)
```bash
./.venv/bin/python run_backtest_cli.py
```

### 4. Avviare la Dashboard Interattiva Streamlit
```bash
./.venv/bin/streamlit run app.py
```
L'interfaccia web si aprirà su `http://localhost:8501`.
