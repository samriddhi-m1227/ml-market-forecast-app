import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional: live watchlist price/beta
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False


# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="Market Open Forecast",
    page_icon="📈",
    layout="wide",
)

# -------------------------
# Theme (light/dark) via CSS
# -------------------------
def inject_theme(dark: bool):
    if dark:
        st.markdown(
            """
<style>
/* ---- Dark theme ---- */
:root, body, .stApp { background: #0b1220 !important; color: #e5e7eb !important; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1,h2,h3,h4 { color: #f9fafb !important; }
.small-note { color: #9ca3af; font-size: 0.95rem; }
hr { margin: 1.2rem 0; border-color: #1f2937; }
.kpi-card { border: 1px solid #1f2937; border-radius: 16px; padding: 14px 16px; background: #0f172a; }
.badge { display:inline-block; padding: 4px 10px; border-radius: 999px; background: #111827; border: 1px solid #1f2937; color: #d1d5db; font-size: 0.85rem; }
.muted { color: #9ca3af; }
.stTabs [data-baseweb="tab"] { color: #d1d5db !important; }
.stTabs [aria-selected="true"] { border-bottom: 2px solid #60a5fa !important; }
.stTextInput input, .stDateInput input, .stNumberInput input { background: #0f172a !important; color: #e5e7eb !important; border: 1px solid #1f2937 !important; }
.stDataFrame { border: 1px solid #1f2937; border-radius: 12px; overflow: hidden; }
</style>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<style>
/* ---- Light theme ---- */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1 { margin-bottom: 0.2rem; }
.small-note { color: #6b7280; font-size: 0.95rem; }
.kpi-card { border: 1px solid #e5e7eb; border-radius: 16px; padding: 14px 16px; background: #ffffff; }
.badge { display:inline-block; padding: 4px 10px; border-radius: 999px; background: #f9fafb; border: 1px solid #e5e7eb; color: #374151; font-size: 0.85rem; }
.muted { color: #6b7280; }
hr { margin: 1.2rem 0; }
</style>
""",
            unsafe_allow_html=True,
        )


# -------------------------
# Paths
# -------------------------
DATA_PATH = os.path.join("data", "feat_sample.csv")
MODEL_PATH = os.path.join("models", "xgboost_final_model.pkl")
FEATS_PATH = os.path.join("models", "model_features.pkl")


# -------------------------
# Cached loaders
# -------------------------
@st.cache_resource
def load_artifacts(model_path: str, feats_path: str):
    model = joblib.load(model_path)
    feature_cols = joblib.load(feats_path)
    return model, feature_cols


@st.cache_data
def load_feat_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    return df


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def get_row_for_date(df: pd.DataFrame, date_val: pd.Timestamp):
    row = df[df["Date"] == date_val]
    if row.empty:
        return None
    return row.iloc[0]


def predict_log_return(model, feature_cols, row_series: pd.Series):
    X_row = pd.DataFrame([{c: row_series.get(c, np.nan) for c in feature_cols}]).fillna(0)
    pred = model.predict(X_row)[0]
    return float(pred), X_row


def implied_next_open(today_open: float, pred_log_ret: float) -> float:
    return float(today_open * np.exp(pred_log_ret))


def baseline_predict_zero(y_true: np.ndarray):
    return np.zeros_like(y_true, dtype=float)


def get_feature_importance(model, feature_cols):
    try:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
        return imp_df
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


@st.cache_data
def fetch_prices_yf(tickers, start=None, end=None, interval="1d"):
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    return data


def compute_beta(asset_rets: pd.Series, market_rets: pd.Series):
    aligned = pd.concat([asset_rets, market_rets], axis=1).dropna()
    if aligned.shape[0] < 10:
        return np.nan
    a = aligned.iloc[:, 0].values
    m = aligned.iloc[:, 1].values
    var_m = np.var(m)
    if var_m == 0:
        return np.nan
    cov_am = np.cov(a, m)[0, 1]
    return float(cov_am / var_m)


# -------------------------
# Sidebar: make it intuitive
# -------------------------
st.sidebar.title("⚙️ Settings")

dark_mode = st.sidebar.toggle("Dark mode", value=True)
inject_theme(dark_mode)

st.sidebar.markdown("**What to predict**")
market_label = st.sidebar.text_input("Market label", value="S&P 500 proxy", help="Display-only label for your market series.")

st.sidebar.markdown("**Pick a date**")
# (We only allow completed dates in the dataset)
# This is more honest than pretending we can do 'today' without full features.
# We'll show a clear note in the UI.
# Load data + artifacts after guardrails.

# -------------------------
# Header
# -------------------------
st.title("📈 Market Open Forecast")
st.markdown(
    """
<div class='small-note'>
<b>What this app does:</b> It predicts tomorrow’s <b>market movement</b> (log return) from market + news features, then converts it into an <b>estimated next-day open</b>.
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("How does it get tomorrow's open? (simple explanation)"):
    st.write(
        "Instead of predicting price directly, the model predicts **how much the market might move** overnight (a log return). "
        "Then we apply that move to **today’s open** to estimate tomorrow’s open:\n\n"
        "Estimated next open = Today open × exp(predicted log return)"
    )
    st.write("We only allow dates that are fully available in the dataset (completed trading days).")


# -------------------------
# Guardrails for missing files
# -------------------------
missing = []
if not os.path.exists(DATA_PATH):
    missing.append(f"- Missing data file: `{DATA_PATH}`")
if not os.path.exists(MODEL_PATH):
    missing.append(f"- Missing model file: `{MODEL_PATH}`")
if not os.path.exists(FEATS_PATH):
    missing.append(f"- Missing features file: `{FEATS_PATH}`")

if missing:
    st.error("Your app files aren’t in place yet:\n\n" + "\n".join(missing))
    st.info(
        "Fix by creating folders `data/` and `models/`, then placing:\n"
        "- `feat_sample.csv` in `data/`\n"
        "- `xgboost_final_model.pkl` + `model_features.pkl` in `models/`"
    )
    st.stop()


# -------------------------
# Load
# -------------------------
df = load_feat_data(DATA_PATH)
model, feature_cols = load_artifacts(MODEL_PATH, FEATS_PATH)

required_cols = set(["Date", "Open"]) | set(feature_cols)
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error("Your feature CSV is missing required columns:\n\n" + "\n".join([f"- {c}" for c in missing_cols]))
    st.stop()

latest_date = df["Date"].max()
min_date = df["Date"].min()

use_latest = st.sidebar.checkbox("Use latest available (recommended)", value=True)
picked_date = st.sidebar.date_input(
    "Date (completed trading days only)",
    value=latest_date.date(),
    min_value=min_date.date(),
    max_value=latest_date.date(),
    help="We only show completed days because today’s full close/news features may not be available yet.",
)
date_ts = latest_date if use_latest else pd.to_datetime(picked_date)

show_advanced = st.sidebar.checkbox("Show advanced details", value=False)

# Freshness badge
st.markdown(
    f"<span class='badge'>Latest available date in dataset: <b>{latest_date.date()}</b></span> "
    f"<span class='badge'>Selected: <b>{date_ts.date()}</b></span>",
    unsafe_allow_html=True,
)

# -------------------------
# Tabs
# -------------------------
tab_forecast, tab_watchlist, tab_backtest, tab_explain = st.tabs(
    ["🔮 Forecast", "📋 Watchlist Impact", "🧪 Backtest", "🧠 Explainability"]
)

# =========================
# TAB 1: FORECAST
# =========================
with tab_forecast:
    st.subheader("Forecast")
    st.caption("Primary output: predicted next-day market move + implied next-day open.")

    row = get_row_for_date(df, date_ts)
    if row is None:
        st.warning("No row found for that date. Try another date.")
        st.stop()

    pred_log_ret, X_row = predict_log_return(model, feature_cols, row)
    today_open = safe_float(row["Open"])
    pred_next_open = implied_next_open(today_open, pred_log_ret)

    # Interpret as %
    pred_pct = (np.exp(pred_log_ret) - 1.0) * 100
    direction = "↑ Up" if pred_pct >= 0 else "↓ Down"

    actual_next_open = safe_float(row["Next_Open"]) if "Next_Open" in df.columns else np.nan
    has_actual = np.isfinite(actual_next_open)

    # KPI row (more intuitive labels)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market", market_label)
    c2.metric("Forecasted move (next day)", f"{pred_pct:+.2f}%  ({direction})")
    c3.metric("Estimated next-day open", f"{pred_next_open:,.2f}")
    c4.metric("Today open (input)", f"{today_open:,.2f}")

    # Accuracy panel (if actual exists)
    if has_actual:
        err = pred_next_open - actual_next_open
        ae = abs(err)
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.write("**Evaluation (from dataset)**")
        st.write(f"- Actual next-day open: **{actual_next_open:,.2f}**")
        st.write(f"- Prediction error: **{err:+.2f}** (abs: **{ae:.2f}**)")
        st.write("<span class='muted'>Note: daily markets are noisy; this app surfaces weak predictive signals.</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Chart
    st.subheader("Open price trend (last 120 trading days)")
    plot_df = df[["Date", "Open"]].dropna().set_index("Date").tail(120)

    fig = plt.figure()
    plt.plot(plot_df.index, plot_df["Open"].values)
    plt.xlabel("Date")
    plt.ylabel("Open")
    plt.title(f"{market_label}: Open history")
    st.pyplot(fig, clear_figure=True)

    st.caption(
        f"Selected date: {date_ts.date()} • Estimated next open: {pred_next_open:,.2f}"
        + (f" • Actual next open: {actual_next_open:,.2f}" if has_actual else "")
    )

    if show_advanced:
        with st.expander("Advanced: model input features (for selected date)"):
            st.dataframe(X_row.T.rename(columns={0: "value"}), use_container_width=True)

        with st.expander("Advanced: raw data row snapshot"):
            st.write(row.to_frame("value"))


# =========================
# TAB 2: WATCHLIST IMPACT
# =========================
with tab_watchlist:
    st.subheader("Watchlist Impact")
    st.caption(
        "Use the market forecast to estimate how your watchlist *might* move, using each ticker’s recent beta vs a market proxy."
    )

    if not YF_OK:
        st.warning("`yfinance` isn't installed. Add it to requirements.txt to enable this tab.")
        st.stop()

    # Market forecast (same date choice)
    row = get_row_for_date(df, date_ts)
    pred_log_ret, _ = predict_log_return(model, feature_cols, row)
    market_pct = (np.exp(pred_log_ret) - 1.0) * 100

    st.markdown(f"<span class='badge'>Market forecast (implied): <b>{market_pct:+.2f}%</b></span>", unsafe_allow_html=True)

    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        tickers_raw = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, NVDA, AMZN, TSLA")
    with colB:
        lookback_days = st.number_input("Beta lookback (days)", min_value=30, max_value=252, value=90, step=10)
    with colC:
        market_proxy = st.text_input("Market proxy", value="SPY")

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    all_tickers = list(dict.fromkeys(tickers + [market_proxy]))

    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(lookback_days * 2))

    data = fetch_prices_yf(all_tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    if data is None or len(data) == 0:
        st.error("Could not fetch prices. Try again or reduce tickers.")
        st.stop()

    open_df = None
    close_df = None

    if isinstance(data.columns, pd.MultiIndex):
        if ("Adj Close" in data.columns.get_level_values(0)):
            close_df = data["Adj Close"].copy()
        elif ("Close" in data.columns.get_level_values(0)):
            close_df = data["Close"].copy()
        open_df = data["Open"].copy() if ("Open" in data.columns.get_level_values(0)) else None
    else:
        close_df = data["Adj Close"].to_frame(all_tickers[0]) if "Adj Close" in data else data["Close"].to_frame(all_tickers[0])
        open_df = data["Open"].to_frame(all_tickers[0]) if "Open" in data else None

    if close_df is None or open_df is None:
        st.error("Missing required price fields from yfinance response.")
        st.stop()

    close_df = close_df.dropna(how="all")
    rets = np.log(close_df / close_df.shift(1))

    if market_proxy not in rets.columns:
        st.error(f"Market proxy `{market_proxy}` not found in fetched returns.")
        st.stop()

    market_rets = rets[market_proxy].dropna()
    latest_opens = open_df.dropna(how="all").iloc[-1].to_dict()

    rows_out = []
    for t in tickers:
        if t not in rets.columns or t not in latest_opens:
            rows_out.append({"Ticker": t, "Today Open": np.nan, "Beta": np.nan, "Implied %": np.nan, "Implied Next Open": np.nan})
            continue

        beta = compute_beta(rets[t], market_rets)
        today_open_t = safe_float(latest_opens.get(t, np.nan))

        implied_log_ret_t = beta * pred_log_ret if np.isfinite(beta) else np.nan
        implied_next_open_t = today_open_t * np.exp(implied_log_ret_t) if np.isfinite(implied_log_ret_t) else np.nan
        implied_pct_t = (np.exp(implied_log_ret_t) - 1.0) * 100 if np.isfinite(implied_log_ret_t) else np.nan

        rows_out.append({
            "Ticker": t,
            "Today Open": today_open_t,
            "Beta": beta,
            "Implied %": implied_pct_t,
            "Implied Next Open": implied_next_open_t,
        })

    out_df = pd.DataFrame(rows_out).sort_values("Implied %", ascending=False)

    avg_move = np.nanmean(out_df["Implied %"].values) if len(out_df) else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Market forecast (implied %)", f"{market_pct:+.2f}%")
    c2.metric("Watchlist average implied move", f"{avg_move:+.2f}%" if np.isfinite(avg_move) else "N/A")
    c3.metric("Proxy used for beta", market_proxy)

    st.divider()
    st.write("### Results (beta-adjusted estimates)")
    st.dataframe(out_df, use_container_width=True)

    st.write("### Visual: implied % moves (dashed line = watchlist average)")
    chart_df = out_df.dropna(subset=["Implied %"]).copy()
    if chart_df.empty:
        st.info("Not enough data to plot implied moves.")
    else:
        fig = plt.figure()
        plt.bar(chart_df["Ticker"], chart_df["Implied %"].values)
        if np.isfinite(avg_move):
            plt.axhline(avg_move, linestyle="--")
        plt.xlabel("Ticker")
        plt.ylabel("Implied % move")
        plt.title("Implied moves (beta-adjusted from market forecast)")
        st.pyplot(fig, clear_figure=True)

    st.caption("Watchlist output is a beta-based approximation (derived), not direct ML per ticker. Educational use only.")


# =========================
# TAB 3: BACKTEST
# =========================
with tab_backtest:
    st.subheader("Backtest")
    st.caption("Evaluates the model on your stored dataset (no live APIs).")

    col1, col2 = st.columns([1, 2])
    with col1:
        backtest_days = st.number_input(
            "Backtest window (last N rows)",
            min_value=50,
            max_value=len(df),
            value=min(400, len(df)),
            step=50
        )
    with col2:
        show_baseline = st.checkbox("Compare to baseline (predict 0 return)", value=True)

    bt_df = df.dropna(subset=["log_ret_t1"]).tail(int(backtest_days)).copy()
    if bt_df.empty:
        st.warning("No backtest rows found (missing log_ret_t1).")
        st.stop()

    X_bt = bt_df[feature_cols].fillna(0)
    y_true = bt_df["log_ret_t1"].values.astype(float)
    y_pred = model.predict(X_bt).astype(float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (log return)", f"{mae:.6f}")
    c2.metric("RMSE (log return)", f"{rmse:.6f}")
    c3.metric("R²", f"{r2:.4f}" if np.isfinite(r2) else "N/A")

    if show_baseline:
        y0 = baseline_predict_zero(y_true)
        mae0 = float(np.mean(np.abs(y_true - y0)))
        rmse0 = float(np.sqrt(np.mean((y_true - y0) ** 2)))
        ss_res0 = float(np.sum((y_true - y0) ** 2))
        r20 = 1.0 - (ss_res0 / ss_tot) if ss_tot != 0 else np.nan

        st.markdown("#### Baseline: predict 0 return")
        b1, b2, b3 = st.columns(3)
        b1.metric("MAE", f"{mae0:.6f}")
        b2.metric("RMSE", f"{rmse0:.6f}")
        b3.metric("R²", f"{r20:.4f}" if np.isfinite(r20) else "N/A")

    st.divider()

    st.write("### Predicted vs Actual (log returns)")
    fig = plt.figure()
    plt.plot(bt_df["Date"].values, y_true, label="Actual")
    plt.plot(bt_df["Date"].values, y_pred, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("log_ret_t1")
    plt.title("Backtest: next-day log return")
    plt.legend()
    st.pyplot(fig, clear_figure=True)

    if "Open" in bt_df.columns:
        implied_open_pred = bt_df["Open"].values.astype(float) * np.exp(y_pred)
        implied_open_actual = bt_df["Open"].values.astype(float) * np.exp(y_true)

        st.write("### Derived Next Open (pred vs actual)")
        fig2 = plt.figure()
        plt.plot(bt_df["Date"].values, implied_open_actual, label="Actual (derived)")
        plt.plot(bt_df["Date"].values, implied_open_pred, label="Predicted (derived)")
        plt.xlabel("Date")
        plt.ylabel("Open")
        plt.title("Next-day open derived from today's open × exp(log return)")
        plt.legend()
        st.pyplot(fig2, clear_figure=True)


# =========================
# TAB 4: EXPLAINABILITY
# =========================
with tab_explain:
    st.subheader("Explainability")
    st.caption("Which inputs mattered most to the XGBoost model?")

    imp_df = get_feature_importance(model, feature_cols)
    if imp_df.empty:
        st.info("Feature importance not available for this model object.")
        st.stop()

    top_n = st.slider("Top features to show", min_value=5, max_value=min(30, len(imp_df)), value=12, step=1)
    imp_top = imp_df.head(top_n).iloc[::-1]

    fig = plt.figure()
    plt.barh(imp_top["feature"], imp_top["importance"].values)
    plt.xlabel("Importance")
    plt.title("Top feature importances (XGBoost)")
    st.pyplot(fig, clear_figure=True)

    with st.expander("Show full importance table"):
        st.dataframe(imp_df, use_container_width=True)

    st.markdown(
        """
**How to read this:** Higher importance means the feature is more often useful for splits across trees.
It does **not** imply causality.
        """
    )


# Footer
st.markdown("---")
st.caption("Educational demo. Not financial advice. • Made by Samriddhi Matharu")
