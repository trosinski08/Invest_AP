"""
app.py
Dashboard Streamlit do monitoringu agenta w czasie rzeczywistym.
Uruchamianie: streamlit run app.py
"""
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import config
from engine.guardrails import CircuitBreaker
from engine.tools import fetch_balance

# ---------------------------------------------------------------------------
# KONFIGURACJA STRONY
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Investment Agent — Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Autonomous AI Investment Agent")
st.caption(f"Para: **{config.TRADING_PAIR}** | Interwał: **{config.TIMEFRAME}** | "
           f"Tryb: **{'PAPER' if config.IS_PAPER_TRADING else '🔴 REAL'}**")

circuit = CircuitBreaker()

# ---------------------------------------------------------------------------
# SIDEBAR — Parametry bezpieczeństwa
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Guardrails")
    st.metric("Max zlecenie", f"${config.MAX_ORDER_VALUE_USD:.0f}")
    st.metric("Max dzienna strata", f"${config.MAX_DAILY_LOSS_USD:.0f}")
    st.metric("Max transakcji/24h", config.MAX_TRADES_PER_24H)
    st.metric("Cooldown", f"{config.COOLDOWN_MINUTES} min")
    st.metric("Stop-Loss", f"{config.STOP_LOSS_PCT}%")
    st.metric("Take-Profit", f"{config.TAKE_PROFIT_PCT}%")

    st.divider()
    st.metric("Model LLM", config.LLM_MODEL)
    st.metric("Pewność min.", f"{config.SENTIMENT_THRESHOLD}")

# ---------------------------------------------------------------------------
# GŁÓWNY PANEL — Stan portfela
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

try:
    balance = fetch_balance()
    daily_pnl = circuit.get_daily_pnl()
    open_pos = circuit.get_open_positions()
    recent_trades = circuit.get_trade_history(hours=24)
    executed = [t for t in recent_trades if t.get("action", "").upper() in ("BUY", "SELL")]

    col1.metric("💰 Saldo (USDT)", f"${balance.get('total_usdt', 0):.2f}")
    col2.metric("📊 Dzienny PnL",
                f"${daily_pnl:+.2f}",
                delta=f"{daily_pnl:+.2f}" if daily_pnl != 0 else None,
                delta_color="normal")
    col3.metric("📂 Otwarte pozycje", f"{len(open_pos)} / {config.MAX_OPEN_POSITIONS}")
    col4.metric("🔄 Transakcje (24h)", f"{len(executed)} / {config.MAX_TRADES_PER_24H}")
except Exception as e:
    st.error(f"Błąd ładowania danych portfela: {e}")
    balance = {}
    daily_pnl = 0

st.divider()

# ---------------------------------------------------------------------------
# OTWARTE POZYCJE
# ---------------------------------------------------------------------------
st.subheader("📂 Otwarte pozycje")
if open_pos:
    pos_df = pd.DataFrame(open_pos)
    st.dataframe(pos_df, use_container_width=True)
else:
    st.info("Brak otwartych pozycji.")

# ---------------------------------------------------------------------------
# HISTORIA TRANSAKCJI
# ---------------------------------------------------------------------------
st.subheader("📜 Historia transakcji (ostatnie 24h)")
if recent_trades:
    trades_df = pd.DataFrame(recent_trades)
    # Formatowanie kolumn
    display_cols = ["timestamp", "action", "pair", "price", "amount",
                    "value_usd", "pnl_usd", "confidence", "reasoning"]
    available_cols = [c for c in display_cols if c in trades_df.columns]
    st.dataframe(
        trades_df[available_cols].sort_values("timestamp", ascending=False),
        use_container_width=True,
        height=400,
    )
else:
    st.info("Brak transakcji w ostatnich 24 godzinach.")

# ---------------------------------------------------------------------------
# PEŁNA HISTORIA
# ---------------------------------------------------------------------------
with st.expander("📋 Pełna historia transakcji"):
    all_trades = circuit.get_all_trades()
    if all_trades:
        full_df = pd.DataFrame(all_trades)
        st.dataframe(full_df, use_container_width=True, height=600)

        # Statystyki
        total_trades = len([t for t in all_trades if t.get("action") in ("BUY", "SELL")])
        total_pnl = sum(t.get("pnl_usd", 0) for t in all_trades)
        wins = len([t for t in all_trades if t.get("pnl_usd", 0) > 0])
        losses = len([t for t in all_trades if t.get("pnl_usd", 0) < 0])

        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric("Łączne transakcje", total_trades)
        scol2.metric("Łączny PnL", f"${total_pnl:+.2f}")
        scol3.metric("Wygrane", wins)
        scol4.metric("Przegrane", losses)
    else:
        st.info("Brak historii transakcji.")

# ---------------------------------------------------------------------------
# CIRCUIT BREAKER STATUS
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🛡️ Status Circuit Breaker")

cb_col1, cb_col2 = st.columns(2)

with cb_col1:
    can_trade, reason = circuit.can_trade(config.MAX_ORDER_VALUE_USD)
    if can_trade:
        st.success("✅ Handel DOZWOLONY")
    else:
        st.error(f"⛔ Handel ZABLOKOWANY: {reason}")

with cb_col2:
    daily_loss_pct = (abs(daily_pnl) / config.MAX_DAILY_LOSS_USD * 100) if daily_pnl < 0 else 0
    st.progress(
        min(daily_loss_pct / 100, 1.0),
        text=f"Dzienna strata: {abs(daily_pnl):.2f} / {config.MAX_DAILY_LOSS_USD:.2f} USD "
             f"({daily_loss_pct:.0f}%)"
    )

# ---------------------------------------------------------------------------
# LOGI AGENTA
# ---------------------------------------------------------------------------
with st.expander("📝 Logi agenta (ostatnie 50 linii)"):
    log_path = config.DATA_DIR / "agent.log"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        st.code("".join(lines[-50:]), language="log")
    else:
        st.info("Brak pliku logów. Uruchom agenta (main.py) aby generować logi.")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ DISCLAIMER: Ten projekt służy wyłącznie celom edukacyjnym. "
    "Handel na rynkach finansowych wiąże się z wysokim ryzykiem utraty kapitału."
)
