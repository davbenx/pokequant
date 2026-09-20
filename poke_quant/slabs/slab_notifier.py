"""
poke_quant/slabs/slab_notifier.py — Formattatore di notifiche e report per Carte Gradate (PSA, BGS, CGC).
Fornisce formattatori per:
  - Notifiche push Telegram in formato Markdown
  - Report visuale da terminale CLI
  - Schede informative per la Dashboard Streamlit
"""

from __future__ import annotations
import datetime
from typing import Dict, List, Any
from poke_quant.slabs.models import SlabSignal, RotationRecommendation, SignalAction


def format_slabs_telegram_alert(scan_results: Dict[str, Any]) -> str:
    """Formatta i segnali su lastre in un messaggio Markdown per Telegram."""
    date_str = scan_results.get("timestamp", datetime.date.today().strftime("%d/%m/%Y"))
    buys: List[SlabSignal] = scan_results.get("buy_signals", [])
    sells: List[SlabSignal] = scan_results.get("sell_signals", [])
    rotations: List[RotationRecommendation] = scan_results.get("rotation_signals", [])
    unverified: List[Dict[str, Any]] = scan_results.get("unverified_or_out_of_stock", [])

    lines = [f"*POKEQUANT · RADAR CARTE GRADATE (SLABS)* 💎\nData: *{date_str}*\n"]

    if not buys and not sells and not rotations and not unverified:
        lines.append("Nessun disallineamento operativo rilevato oggi.")
        lines.append("Tutti gli spread PSA/BGS/CGC e le Pop velocity rimangono nella fascia di equilibrio.")
        return "\n".join(lines)

    # 1. SEGNALI BUY VERIFICATI
    if buys:
        lines.append(f"🟢 *OPPORTUNITÀ D'ACQUISTO VERIFICATE (BUY)* [{len(buys)}]")
        for b in buys:
            target_g = b.target_grade.value if hasattr(b.target_grade, "value") else str(b.target_grade)
            avail_tag = f"✅ Inserzione Reale: {b.seller_country} ({b.active_listing_count} disp.)"
            link_tag = f"[🛒 Acquista su Cardmarket]({b.cardmarket_direct_url})" if b.cardmarket_direct_url else ""
            lines.append(
                f"• *{b.card_name}*\n"
                f"  Target: *{target_g}*\n"
                f"  Prezzo Attuale: *{b.current_price_eur:.1f} €* | Fair Value: *{b.fair_value_eur:.1f} €*\n"
                f"  Margine di Sicurezza: *+{b.margin_of_safety_pct:.1f}%*\n"
                f"  Disponibilità: _{avail_tag}_\n"
                f"  {link_tag}\n"
                f"  Edge: _{b.primary_edge.value}_\n"
                f"  {b.reason}\n"
            )

    # 2. OPPORTUNITÀ TEORICHE FUORI MERCATO / OUT OF STOCK
    if unverified:
        lines.append(f"⚠️ *COMP TEORICI FUORI MERCATO / OOS* [{len(unverified)}]")
        for u in unverified:
            link_md = f"[Vedi su Cardmarket]({u['direct_url']})" if u.get('direct_url') else ""
            lines.append(
                f"• *{u['name']}* ({u['grade']})\n"
                f"  Target: *{u['target_price']:,.0f} €* | Min Ask Reale: *{u['lowest_active_ask']:,.0f} €*\n"
                f"  _Attualmente esaurita sul mercato secondario europeo al target._ {link_md}\n"
            )

    # 3. SEGNALI SELL / TAKE-PROFIT
    if sells:
        lines.append(f"🔴 *PRESE DI BENEFICIO / ALLERTE DILUIZIONE (SELL)* [{len(sells)}]")
        for s in sells:
            grade_str = s.target_grade.value if hasattr(s.target_grade, "value") else str(s.target_grade)
            lines.append(
                f"• *{s.card_name}* ({grade_str})\n"
                f"  Prezzo Mercato: *{s.current_price_eur:.1f} €*\n"
                f"  Trigger: _{s.primary_edge.value}_\n"
                f"  {s.reason}\n"
            )

    # 4. RACCOMANDAZIONI DI ROTAZIONE
    if rotations:
        lines.append(f"🔄 *ROTAZIONE OTTIMIZZATA DEL CAPITALE* [{len(rotations)}]")
        for r in rotations:
            lines.append(
                f"• *VENDI*: {r.holding_card_name} ({r.holding_grade.value})\n"
                f"  Prezzo: {r.holding_current_price:.1f} € (ROI attuale: +{r.holding_unrealized_roi_pct:.1f}%)\n"
                f"• *COMPRA*: {r.target_card_name} ({r.target_grade.value})\n"
                f"  Prezzo: {r.target_current_price:.1f} € (Sconto vs Fair Value: +{r.target_margin_of_safety_pct:.1f}%)\n"
                f"• *Alpha Differenziale Netto*: *+{r.net_alpha_differential_pct:.1f}%*\n"
                f"  {r.rationale}\n"
            )

    return "\n".join(lines)


def print_slabs_cli_summary(scan_results: Dict[str, Any]):
    """Stampa un report da riga di comando pulito ed elegante."""
    print("\n" + "=" * 78)
    print("  💎 POKEQUANT — RADAR STRATEGICO CARTE GRADATE (PSA · BGS · CGC)")
    print("=" * 78)

    buys = scan_results.get("buy_signals", [])
    sells = scan_results.get("sell_signals", [])
    rotations = scan_results.get("rotation_signals", [])
    rejected = scan_results.get("rejected_controls", [])
    unverified = scan_results.get("unverified_or_out_of_stock", [])

    print(f"📊 Universo Scansionato: {scan_results.get('total_universe_scanned', 0)} carte monitorate")
    print(f"🛡️  Asset Controllo Rigettati (Anti-Survivorship): {len(rejected)}")
    print(f"🟢 Opportunità BUY Eseguibili (Verificate): {len(buys)}")
    print(f"⚠️  Opportunità Teoriche Fuori Mercato / OOS: {len(unverified)}")
    print(f"🔴 Prese di Beneficio SELL: {len(sells)}")
    print(f"🔄 Rotazioni Consigliate: {len(rotations)}")
    print("-" * 78)

    if buys:
        print("\n🟢 SEGNALI BUY CON EDGE MATEMATICO & INSERZIONE ATTIVA VERIFICATA:")
        for idx, b in enumerate(buys, 1):
            g_str = b.target_grade.value if hasattr(b.target_grade, "value") else str(b.target_grade)
            print(f"  [{idx}] {b.card_name} — {g_str}")
            print(f"      Prezzo: {b.current_price_eur:.2f} € | Fair Value: {b.fair_value_eur:.2f} € | Margine: +{b.margin_of_safety_pct:.1f}%")
            print(f"      Disponibilità: ✅ REALE ({b.seller_country}, {b.active_listing_count} unità su Cardmarket)")
            if b.cardmarket_direct_url:
                print(f"      Link Diretto: {b.cardmarket_direct_url}")
            print(f"      Edge: {b.primary_edge.value}")
            print(f"      Motivazione: {b.reason}\n")

    if unverified:
        print("\n⚠️ OPPORTUNITÀ TEORICHE FUORI MERCATO / OUT-OF-STOCK (NON ACQUISTABILI AL TARGET):")
        for idx, u in enumerate(unverified, 1):
            print(f"  [{idx}] {u['name']} — {u['grade']}")
            print(f"      Prezzo Target: {u['target_price']:,.0f} € vs Ask Minimo Reale su Cardmarket: {u['lowest_active_ask']:,.0f} €")
            print(f"      Stato: {u['status']} — {u['reason']}")
            if u.get('direct_url'):
                print(f"      Link Cardmarket: {u['direct_url']}")
            print()

    if sells:
        print("\n🔴 SEGNALI SELL / TAKE-PROFIT:")
        for idx, s in enumerate(sells, 1):
            g_str = s.target_grade.value if hasattr(s.target_grade, "value") else str(s.target_grade)
            print(f"  [{idx}] {s.card_name} — {g_str}")
            print(f"      Prezzo di Liquidazione: {s.current_price_eur:.2f} €")
            print(f"      Motivazione: {s.reason}\n")

    if rotations:
        print("\n🔄 RACCOMANDAZIONI DI ROTAZIONE DEL CAPITALE:")
        for idx, r in enumerate(rotations, 1):
            print(f"  [{idx}] RUOTA DA: {r.holding_card_name} ➔ A: {r.target_card_name}")
            print(f"      Alpha Differenziale Netto: +{r.net_alpha_differential_pct:.1f}% (netto fee e spedizioni)")
            print(f"      Motivazione: {r.rationale}\n")

    print("=" * 78 + "\n")
