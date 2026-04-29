"""
app.py
Streamlit Dashboard for real-time agent monitoring.
Run: streamlit run app.py
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
st.caption(f"Pair: **{config.TRADING_PAIR}** | Interval: **{config.TIMEFRAME}** | "
           f"Mode: **{'PAPER' if config.IS_PAPER_TRADING else '🔴 REAL'}**")

circuit = CircuitBreaker()

# ---------------------------------------------------------------------------
# SIDEBAR — Safety Parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Guardrails")
    st.metric("Max Order", f"${config.MAX_ORDER_VALUE_USD:.0f}")
    st.metric("Max Daily Loss", f"${config.MAX_DAILY_LOSS_USD:.0f}")
    st.metric("Max Trades/24h", config.MAX_TRADES_PER_24H)
    st.metric("Cooldown", f"{config.COOLDOWN_MINUTES} min")
    st.metric("Stop-Loss", f"{config.STOP_LOSS_PCT}%")
    st.metric("Take-Profit", f"{config.TAKE_PROFIT_PCT}%")

    st.divider()
    st.metric("LLM Model", config.LLM_MODEL)
    st.metric("Min Confidence", f"{config.SENTIMENT_THRESHOLD}")

# ---------------------------------------------------------------------------
# MAIN PANEL — Portfolio Status
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

try:
    balance = fetch_balance()
    daily_pnl = circuit.get_daily_pnl()
    open_pos = circuit.get_open_positions()
    recent_trades = circuit.get_trade_history(hours=24)
    executed = [t for t in recent_trades if t.get("action", "").upper() in ("BUY", "SELL")]

    col1.metric("💰 Balance (USDT)", f"${balance.get('total_usdt', 0):.2f}")
    col2.metric("📊 Daily PnL",
                f"${daily_pnl:+.2f}",
                delta=f"{daily_pnl:+.2f}" if daily_pnl != 0 else None,
                delta_color="normal")
    col3.metric("📂 Open Positions", f"{len(open_pos)} / {config.MAX_OPEN_POSITIONS}")
    col4.metric("🔄 Trades (24h)", f"{len(executed)} / {config.MAX_TRADES_PER_24H}")
except Exception as e:
    st.error(f"Error loading portfolio data: {e}")
    balance = {}
    daily_pnl = 0

st.divider()

# ---------------------------------------------------------------------------
# OPEN POSITIONS
# ---------------------------------------------------------------------------
st.subheader("📂 Open Positions")
if open_pos:
    pos_df = pd.DataFrame(open_pos)
    st.dataframe(pos_df, use_container_width=True)
else:
    st.info("No open positions.")

# ---------------------------------------------------------------------------
# TRADE HISTORY
# ---------------------------------------------------------------------------
st.subheader("📜 Trade History (Last 24h)")
if recent_trades:
    trades_df = pd.DataFrame(recent_trades)
    # Format columns
    display_cols = ["timestamp", "action", "pair", "price", "amount",
                    "value_usd", "pnl_usd", "confidence", "reasoning"]
    available_cols = [c for c in display_cols if c in trades_df.columns]
    st.dataframe(
        trades_df[available_cols].sort_values("timestamp", ascending=False),
        use_container_width=True,
        height=400,
    )
else:
    st.info("No trades in the last 24 hours.")

# ---------------------------------------------------------------------------
# FULL HISTORY
# ---------------------------------------------------------------------------
with st.expander("📋 Full Trade History"):
    all_trades = circuit.get_all_trades()
    if all_trades:
        full_df = pd.DataFrame(all_trades)
        st.dataframe(full_df, use_container_width=True, height=600)

        # Statistics
        total_trades = len([t for t in all_trades if t.get("action") in ("BUY", "SELL")])
        total_pnl = sum(t.get("pnl_usd", 0) for t in all_trades)
        wins = len([t for t in all_trades if t.get("pnl_usd", 0) > 0])
        losses = len([t for t in all_trades if t.get("pnl_usd", 0) < 0])

        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric("Total Trades", total_trades)
        scol2.metric("Total PnL", f"${total_pnl:+.2f}")
        scol3.metric("Wins", wins)
        scol4.metric("Losses", losses)
    else:
        st.info("No trade history.")

# ---------------------------------------------------------------------------
# CIRCUIT BREAKER STATUS
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🛡️ Circuit Breaker Status")

cb_col1, cb_col2 = st.columns(2)

with cb_col1:
    can_trade, reason = circuit.can_trade(config.MAX_ORDER_VALUE_USD)
    if can_trade:
        st.success("✅ Trading ALLOWED")
    else:
        st.error(f"⛔ Trading BLOCKED: {reason}")

with cb_col2:
    daily_loss_pct = (abs(daily_pnl) / config.MAX_DAILY_LOSS_USD * 100) if daily_pnl < 0 else 0
    st.progress(
        min(daily_loss_pct / 100, 1.0),
        text=f"Daily Loss: {abs(daily_pnl):.2f} / {config.MAX_DAILY_LOSS_USD:.2f} USD "
             f"({daily_loss_pct:.0f}%)"
    )

# ---------------------------------------------------------------------------
# AGENT LOGS
# ---------------------------------------------------------------------------
with st.expander("📝 Agent Logs (Last 50 lines)"):
    log_path = config.DATA_DIR / "agent.log"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        st.code("".join(lines[-50:]), language="log")
    else:
        st.info("No log file found. Run agent (main.py) to generate logs.")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ DISCLAIMER: This project is for educational purposes only. "
    "Trading financial markets carries high risk of capital loss."
)
