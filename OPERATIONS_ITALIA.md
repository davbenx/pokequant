# Runbook Operativo — Esecuzione dall'Italia

Guida pratica per usare il segnale validato (`scripts/run_monthly_production_signal.py`)
nella vita reale, operando dall'Italia. Non è un consiglio finanziario o fiscale —
è la traduzione dei risultati quantitativi in passi concreti, con i limiti espliciti.

## 0. Solo box sigillati — le singole gradate sono sospese

Questo runbook copre **solo TS Momentum su box sigillati** (DSR 0.913, validato).
Le singole gradate sono state chiuse dopo una ricerca sistematica su 5 famiglie di
fattori (età/carry, TS momentum, cross-sectional momentum, dip mean-reversion, rarità
ex-ante) sull'universo reale e bias-auditato (928 carte): **nessuna supera la soglia
istituzionale** (`scripts/optimize_and_falsify.py::section_singles_factor_search`).
Il candidato migliore (dip mean-reversion) aveva PBO=0.514 e Sharpe di segno opposto
tra prima e seconda metà del campione storico — non un fattore stabile, solo un
regime di mercato specifico. `scripts/generate_carry_signal_singles.py` resta
disponibile per ricerca, ma non genera più segnali operativi finché non emerge
un fattore che passi DSR/PBO/bootstrap/walk-forward. Dettagli nello stato di
validazione documentato in ciascun file sotto `poke_quant/engine/strategies/`.

## 1. Cosa fa il sistema automaticamente

Il 2 di ogni mese (`.github/workflows/monthly_signal.yml`):
1. Ricostruisce lo storico prezzi da PriceCharting con tasso EUR/USD reale del mese.
2. Ri-applica il filtro di attendibilità (esclude serie con salti di prezzo implausibili).
3. Genera il segnale **TS Momentum** su box sigillati era 2019+
   (`scripts/generate_monthly_signal.py`) — unico segnale operativo, vedi punto 0.
4. Invia un riepilogo su Telegram (se configurati `TELEGRAM_TOKEN`/`TELEGRAM_CHAT_ID`
   nei secret del repository GitHub — Settings → Secrets and variables → Actions).
5. Pusha i dati aggiornati sul repo, con la suite di test come sanity check prima del push.

**Cosa NON fa**: non verifica se il prodotto è davvero disponibile, non verifica il
prezzo eseguibile reale, non esegue nessun acquisto/vendita. Genera una lista, tu
decidi ed esegui a mano.

## 2. Il gate di liquidità (obbligatorio, non opzionale)

Prima di comprare qualunque cosa in lista:
1. Controlla il prezzo reale su Cardmarket (il tuo mercato principale, operando
   dall'Italia) per quell'item specifico.
2. Registra l'osservazione con `python scripts/log_execution_price.py <item_id> --ask <prezzo_visto>`.
3. Se lo scarto dashboard-vs-reale è enorme o il prodotto non è disponibile,
   **non forzare l'acquisto** solo perché il modello lo segnala — il modello non sa
   che non è disponibile.
4. Fonte compliant per automatizzare questo in futuro: API ufficiale eBay (Browse API,
   registrazione gratuita developer.ebay.com) o API MKM Cardmarket se disponibile.
   Non usare servizi terzi che scrappano Cardmarket (vedi discussione precedente sul
   perché) né bypassare protezioni anti-bot.

## 3. Sourcing — priorità dei canali

1. **Cardmarket** — priorità assoluta per singole gradate e box EU-native. Fee 5%,
   nessuna dogana intra-UE.
2. **eBay.it / eBay.de** — secondo canale, specialmente per box sigillati USA/JP dove
   Cardmarket ha meno offerta. Attenzione a dogana/IVA se il venditore è extra-UE
   (USA, UK post-Brexit, Giappone) — puoi arrivare a pagare IVA 22% + eventuali dazi
   in più rispetto al prezzo di listino.
3. **TCGplayer / venditori USA diretti** — solo se il differenziale di prezzo supera
   abbondantemente il costo di importazione (dogana + spedizione + tempo). Il progetto
   NON modella questo costo automaticamente — calcolalo a mano caso per caso.

## 4. Dimensionamento delle posizioni

Le strategie hanno già un cap di allocazione integrato (`max_allocation_pct` nel
codice, tipicamente 12-15% del capitale per singola posizione). Non superarlo
manualmente. Regola aggiuntiva raccomandata (non ancora automatizzata):
- Pesa MENO le posizioni sui set/carte uscite negli ultimi 12-18 mesi — è il fatto più
  riprodotto in questa sessione di validazione (drawdown molto peggiore sulla coorte
  più giovane, confermato 3 volte su dataset indipendenti).

## 5. Fiscalità — aperto, non risolto

Il motore non applica alcuna imposta ai rendimenti mostrati. Il trattamento fiscale
delle vendite occasionali di collezionabili in Italia non è stato chiarito in questa
sessione (redditi diversi finanziari al 26% vs vendita occasionale non tassabile sono
letture diverse, e non sono qualificato a stabilire quale si applica al tuo caso).
**Consulta un fiscalista prima di trattare questi numeri come rendimento netto reale.**

## 6. Quando fermarsi / rivalidare

- Rilancia `scripts/optimize_and_falsify.py` almeno ogni 6 mesi: se il PBO sale sopra
  il 50% o il DSR scende sotto ~0.7, il segnale ha perso l'evidenza statistica che
  lo supportava oggi.
- Per riattivare le singole: non basta ri-eseguire gli stessi 5 fattori già falliti.
  Serve un fattore nuovo (es. dati pop report reali quando/se disponibili con storico
  point-in-time, non lo snapshot corrente) che passi DSR ≥ ~0.90, PBO ≤ ~25-30% E
  regga lo split walk-forward H1/H2 senza invertire segno — lo stesso standard usato
  per TS Momentum sealed.
- Se una singola posizione supera il cap di allocazione per apprezzamento, valuta un
  ribilanciamento (vendita parziale) — non è automatizzato, va deciso a mano.
- Se PriceCharting cambia struttura di pagina (rischio reale: già successo per
  `scarlet_violet_base_bb`, slug rotto), gli script di fetch falliranno in modo
  visibile (loggano l'errore, non inventano dati) — non ignorare quei log.

## 7. Cosa manca ancora prima di allocare capitale reale

1. **Liquidità reale verificata** — punto 2 sopra, il gap più importante.
2. Conferma fiscale — punto 5.
3. Un periodo di osservazione "paper" (segnali generati ma non eseguiti) per almeno
   2-3 cicli mensili, per vedere se il segnale si comporta come nel backtest prima
   di impegnare capitale vero.
