"""
Streamlit-App: Optimales SMI-Portfolio – Offline, ohne yfinance/scipy
Autor: ChatGPT (für Matthias)

Features (Cloud‑tauglich):
- CSV‑Upload (Preise oder Renditen) ODER synthetische SMI‑Demo‑Daten
- Long‑only, Gewichtslimit (Cap), Max‑Sharpe, GMV, Ziel‑Rendite, Efficient Frontier
- Visualisierung mit matplotlib
- Template‑Generator (alle 20 SMI‑Ticker), Demo‑CSV-Export
- Sanity‑Checks (Sum(w)=1, Long‑only, PSD‑Kovarianz, μ‑Plausibilität, NaN‑Kontrolle)

Start lokal:  streamlit run app.py
"""

from typing import Optional, Tuple
from datetime import date
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================
# Globale Konfiguration
# =========================================

st.set_page_config(page_title="Optimales SMI-Portfolio (offline)", layout="wide")

SMI_TICKERS = [
    "NESN.SW","NOVN.SW","ROG.SW","UBSG.SW","ZURN.SW","ABBN.SW","SIKA.SW","GIVN.SW",
    "LONN.SW","SREN.SW","SGSN.SW","HOLN.SW","GEBN.SW","UHR.SW","CFR.SW","ALCN.SW",
    "LOGN.SW","SOON.SW","EMSN.SW","BALN.SW"
]

# =========================================
# Mathe/Optimizer – nur NumPy (kein scipy)
# =========================================

def _normalize_sum1(w: np.ndarray) -> np.ndarray:
    s = w.sum()
    return w if s == 0 else (w / s)


def _project_capped_simplex(y: np.ndarray, cap: float = 1.0, s: float = 1.0) -> np.ndarray:
    """Projektion auf {w: sum(w)=s, 0<=w<=cap} via Bisection."""
    lo = (y - cap).min() - 1.0
    hi = y.max() + 1.0
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        w = np.clip(y - lam, 0.0, cap)
        if w.sum() > s:
            lo = lam
        else:
            hi = lam
    return w


def _pg_min_var(mu: np.ndarray, cov: np.ndarray, target_ret: Optional[float], long_only: bool, w_cap: Optional[float], max_iter: int = 2000) -> np.ndarray:
    n = len(mu)
    w = np.full(n, 1.0 / n)
    C = cov
    cap = 1.0 if (w_cap is None) else float(w_cap)
    beta = 1000.0 if target_ret is not None else 0.0  # Penalty für w·μ ≥ target
    tr = target_ret if target_ret is not None else -1e9
    t = 0.1
    for _ in range(max_iter):
        Cw = C @ w
        pen = max(0.0, tr - float(w @ mu))
        grad = 2.0 * Cw - (2.0 * beta * pen) * mu
        y = w - t * grad
        if long_only:
            w_new = _project_capped_simplex(y, cap=cap, s=1.0)
        else:
            w_new = _normalize_sum1(y)
        old_obj = float(w @ C @ w)
        new_obj = float(w_new @ C @ w_new)
        if new_obj > old_obj + 1e-12 and t > 1e-6:
            t *= 0.5
            continue
        w = w_new
        if np.linalg.norm(grad) < 1e-6:
            break
    return _normalize_sum1(np.clip(w, 0, 1)) if long_only else _normalize_sum1(w)


def _pg_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float, long_only: bool, w_cap: Optional[float], max_iter: int = 3000) -> np.ndarray:
    n = len(mu)
    w = np.full(n, 1.0 / n)
    C = cov
    cap = 1.0 if (w_cap is None) else float(w_cap)
    t = 0.05
    eps = 1e-12
    for _ in range(max_iter):
        Cw = C @ w
        a = float(w @ mu) - rf
        b2 = float(w @ Cw) + eps
        b = np.sqrt(b2)
        # Grad von -Sharpe = -(μ)/b + a*(Cw)/b^3
        grad = -(mu) / b + (a * Cw) / (b ** 3)
        y = w - t * grad
        if long_only:
            w = _project_capped_simplex(y, cap=cap, s=1.0)
        else:
            w = _normalize_sum1(y)
        t = min(0.2, t * 1.01)
    return _normalize_sum1(np.clip(w, 0, 1)) if long_only else _normalize_sum1(w)


def solve_min_var(mu: np.ndarray, cov: np.ndarray, long_only: bool = True, target_ret: Optional[float] = None, w_max: Optional[float] = None):
    w = _pg_min_var(mu, cov, target_ret=target_ret, long_only=long_only, w_cap=w_max)
    return w


def solve_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, long_only: bool = True, w_max: Optional[float] = None):
    w = _pg_max_sharpe(mu, cov, rf=rf, long_only=long_only, w_cap=w_max)
    return w


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int, long_only: bool, w_cap: Optional[float]):
    m = np.asarray(mu)
    C = np.asarray(cov)
    r_min, r_max = float(m.min()), float(m.max())
    targets = np.linspace(r_min, r_max, n_points)
    rets, vols, ws = [], [], []
    for tr in targets:
        try:
            w = _pg_min_var(m, C, target_ret=tr, long_only=long_only, w_cap=w_cap)
            rets.append(float(w @ m))
            vols.append(float(np.sqrt(w @ C @ w)))
            ws.append(w)
        except Exception:
            continue
    return np.array(rets), np.array(vols), np.array(ws)

# =========================================
# Daten-Utilities
# =========================================

def annualize_stats(returns: pd.DataFrame, freq: str) -> Tuple[pd.Series, pd.DataFrame, int]:
    freq_map = {"Daily": 252, "Weekly": 52, "Monthly": 12}
    periods = freq_map.get(freq, 252)
    mu = returns.mean() * periods
    cov = returns.cov() * periods
    return mu, cov, periods


def resample_prices(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "Daily":
        return prices
    elif freq == "Weekly":
        return prices.resample("W-FRI").last()
    else:
        return prices.resample("M").last()


def to_log_returns_from_prices(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    px = resample_prices(prices, freq)
    px = px.mask(px <= 0)
    rets = np.log(px / px.shift(1))
    return rets.dropna(how="all")


def winsorize_df(df: pd.DataFrame, p: float = 0.01) -> pd.DataFrame:
    low = df.quantile(p)
    high = df.quantile(1 - p)
    return df.clip(lower=low, upper=high, axis=1)


def parse_uploaded_csv(file) -> pd.DataFrame:
    raw = pd.read_csv(file)
    date_col = None
    for cand in ["Date", "date", "Datum", "Time", "Datetime"]:
        if cand in raw.columns:
            date_col = cand
            break
    if date_col is None:
        date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df = raw.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not df.index.is_unique:
        df = df.groupby(df.index).last()
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return df


def make_csv_template(start: pd.Timestamp, end: pd.Timestamp, freq: str = "Daily") -> pd.DataFrame:
    if freq == "Daily":
        idx = pd.bdate_range(start=start, end=end)
    elif freq == "Weekly":
        idx = pd.date_range(start=start, end=end, freq="W-FRI")
    else:
        idx = pd.date_range(start=start, end=end, freq="M")
    df = pd.DataFrame(index=idx, columns=SMI_TICKERS, dtype=float)
    df.index.name = "Date"
    return df.reset_index()


def demo_prices(seed: int = 42, n_years: int = 6, freq: str = "Daily") -> pd.DataFrame:
    np.random.seed(seed)
    periods_map = {"Daily": 252, "Weekly": 52, "Monthly": 12}
    periods = periods_map[freq]
    T = n_years * periods
    n_assets = len(SMI_TICKERS)
    mu_annual = np.linspace(0.04, 0.10, n_assets)
    sigma_annual = np.linspace(0.15, 0.25, n_assets)
    corr = 0.3 + 0.5 * np.ones((n_assets, n_assets))
    np.fill_diagonal(corr, 1.0)
    sigma = sigma_annual / np.sqrt(periods)
    cov = np.outer(sigma, sigma) * corr
    L = np.linalg.cholesky(cov)
    z = np.random.randn(T, n_assets) @ L.T
    drift = (mu_annual / periods) - 0.5 * np.diag(cov)
    log_ret = drift + z
    start_prices = 100 * (1 + 0.1 * np.random.rand(n_assets))
    log_price = np.cumsum(log_ret, axis=0) + np.log(start_prices)
    prices = np.exp(log_price)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T)
    df = pd.DataFrame(prices, index=idx, columns=SMI_TICKERS)
    if freq != "Daily":
        df = resample_prices(df, freq)
    return df

# =========================================
# UI – Seitenleiste
# =========================================

st.title("📈 Optimales Portfolio für den SMI – Offline Efficient Frontier")
st.caption("Mean‑Variance‑Optimierung ohne yfinance/scipy/matplotlib. CSV oder Demo‑Daten.")

with st.sidebar:
    st.header("1) Datenquelle")
    data_mode = st.radio("Quelle", ["CSV hochladen", "Demo‑Daten (synthetisch)"])

    if data_mode == "CSV hochladen":
        st.markdown("**CSV‑Schema (Wide):** Erste Spalte = Datum, weitere Spalten = Ticker. Werte = Close ODER Returns.")
        uploaded = st.file_uploader("CSV hochladen", type=["csv"])
        st.subheader("CSV‑Interpretation")
        csv_kind = st.radio("Inhalt der CSV", ["Preise (Close)", "Renditen"], index=0)
        if csv_kind == "Renditen":
            ret_kind = st.radio("Skalierung der Renditen", ["Dezimal (0.01=1%)", "Prozent (1=1%)"], index=0)
            treat_as_log = st.checkbox("Bereits Log‑Returns", value=False)
        else:
            fill_choice = st.selectbox("Fehlende Werte", ["Keine Füllung", "Vorwärts/Zurück füllen"], index=1)
            zero_to_nan = st.checkbox("Nicht‑positive Preise als fehlend behandeln", value=True)

        st.divider()
        st.subheader("Template/Demo‑Exports")
        t_start = st.date_input("Template Start", value=date(2018, 1, 1))
        t_end = st.date_input("Template Ende", value=date.today())
        t_freq = st.selectbox("Template Frequenz", ["Daily", "Weekly", "Monthly"], index=0)
        if st.button("📄 CSV‑Template (alle SMI‑Ticker) erzeugen"):
            tpl = make_csv_template(pd.to_datetime(t_start), pd.to_datetime(t_end), t_freq)
            st.session_state["tpl_csv"] = tpl.to_csv(index=False).encode("utf-8")
        if "tpl_csv" in st.session_state:
            st.download_button("📥 Template herunterladen", data=st.session_state["tpl_csv"], file_name="SMI_template.csv", mime="text/csv")
    else:
        demo_freq = st.selectbox("Demo‑Frequenz", ["Daily", "Weekly", "Monthly"], index=0)
        demo_years = st.slider("Jahre Historie", 3, 12, 6)
        if st.button("📄 Demo‑CSV generieren"):
            demo_df = demo_prices(n_years=demo_years, freq=demo_freq)
            demo_csv = demo_df.reset_index().rename(columns={"index": "Date"})
            demo_csv.index.name = None
            st.session_state["demo_csv"] = demo_csv.to_csv(index=False).encode("utf-8")
        if "demo_csv" in st.session_state:
            st.download_button("📥 Demo‑CSV herunterladen", data=st.session_state["demo_csv"], file_name="SMI_demo_prices.csv", mime="text/csv")

    st.divider()
    st.header("2) Statistik & Robustheit")
    freq = st.selectbox("Frequenz (für Returns)", ["Daily", "Weekly", "Monthly"], index=0)
    use_winsor = st.checkbox("Extremwerte winsorisieren (1%/99%)", value=False)

    st.divider()
    st.header("3) Optimierung")
    long_only = st.checkbox("Long‑only (keine Shorts)", value=True)
    w_cap = st.slider("Max. Gewicht pro Asset", 0.0, 1.0, 0.15, 0.01)
    rf = st.number_input("Risikofreier Zins p.a.", value=0.005, step=0.001, format="%.3f")
    n_frontier = st.slider("Punkte auf der Frontier", 10, 200, 60, 5)

    st.divider()
    st.header("4) Portfolio‑Auswahl")
    choice = st.radio("Typ", ["Tangency (max. Sharpe)", "GMV (Min‑Var)", "Ziel‑Rendite"], index=0)
    target_ret = None
    if choice == "Ziel‑Rendite":
        target_ret = st.number_input("Zielrendite p.a.", value=0.06, step=0.005, format="%.3f")

# =========================================
# Daten laden & Returns ableiten
# =========================================

if data_mode == "CSV hochladen":
    if uploaded is None:
        st.info("Bitte CSV hochladen – oder links auf Demo‑Daten umstellen.")
        st.stop()
    df_in = parse_uploaded_csv(uploaded)
    st.subheader("CSV‑Diagnose")
    st.write(f"Zeilen: {df_in.shape[0]}, Spalten: {df_in.shape[1]}")
    st.dataframe(df_in.tail())
    st.write("Anteil fehlender Werte je Spalte:")
    st.dataframe(df_in.isna().mean().round(3).to_frame("missing_ratio"))

    if csv_kind == "Preise (Close)":
        px = df_in.copy()
        if 'zero_to_nan' in locals() and zero_to_nan:
            px = px.mask(px <= 0)
        if 'fill_choice' in locals() and fill_choice == "Vorwärts/Zurück füllen":
            px = px.ffill().bfill()
        returns = to_log_returns_from_prices(px, freq)
    else:
        rets = df_in.copy()
        if 'ret_kind' in locals() and ret_kind == "Prozent (1=1%)":
            rets = rets / 100.0
        if 'treat_as_log' in locals() and treat_as_log:
            returns = rets
        else:
            returns = np.log1p(rets)
else:
    prices = demo_prices(n_years=demo_years, freq=freq)
    st.subheader("Demo‑Daten (Preise)")
    st.write(f"Zeilen: {prices.shape[0]}, Spalten: {prices.shape[1]}")
    st.dataframe(prices.tail())
    returns = to_log_returns_from_prices(prices, freq)

if use_winsor:
    returns = winsorize_df(returns, 0.01)

mu, cov, periods = annualize_stats(returns, freq)

# =========================================
# Optimierung & Frontier
# =========================================

try:
    if choice == "Tangency (max. Sharpe)":
        w_opt = solve_max_sharpe(mu.values, cov.values, rf=rf, long_only=long_only, w_max=w_cap if long_only else None)
    elif choice == "GMV (Min‑Var)":
        w_opt = solve_min_var(mu.values, cov.values, long_only=long_only, target_ret=None, w_max=w_cap if long_only else None)
    else:
        w_opt = solve_min_var(mu.values, cov.values, long_only=long_only, target_ret=target_ret, w_max=w_cap if long_only else None)
except Exception as e:
    st.error(f"Optimierung fehlgeschlagen: {e}")
    st.stop()

w_opt = pd.Series(np.clip(w_opt, 0, 1), index=mu.index)
w_opt = w_opt / w_opt.sum()

p_ret = float(w_opt @ mu)
p_vol = float(np.sqrt(w_opt @ cov @ w_opt))
p_sharpe = (p_ret - rf) / p_vol if p_vol > 0 else np.nan

with st.spinner("Berechne Efficient Frontier …"):
    ef_rets, ef_vols, ef_ws = efficient_frontier(mu.values, cov.values, n_points=n_frontier, long_only=long_only, w_cap=w_cap if long_only else None)

# =========================================
# Visualisierung – matplotlib
# =========================================

st.subheader("Efficient Frontier")
fig = plt.figure(figsize=(8, 6))
plt.scatter(ef_vols, ef_rets, s=12, label="Efficient Frontier")
plt.scatter([p_vol], [p_ret], s=120, marker="*", label=f"Ausgewählt ({choice})")
plt.xlabel("Volatilität (σ)")
plt.ylabel("Erwartete Rendite (μ)")
plt.legend()
st.pyplot(fig)

# =========================================
# Outputs
# =========================================

st.subheader("Optimale Gewichte")
weights_df = pd.DataFrame({"Weight": w_opt.round(6)}).sort_values("Weight", ascending=False)
st.dataframe(weights_df)

st.markdown(
    f"**Kennzahlen:**  "
    + f"Rendite p.a.: **{p_ret:.2%}**, Volatilität p.a.: **{p_vol:.2%}**, Sharpe: **{p_sharpe:.2f}** (rf={rf:.2%})"
)

# Downloads
csv_bytes = weights_df.to_csv().encode("utf-8")
st.download_button("📥 Gewichte als CSV", data=csv_bytes, file_name="smi_portfolio_weights.csv", mime="text/csv")

frontier_df = pd.DataFrame({"vol": ef_vols, "ret": ef_rets})
frontier_csv = frontier_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Frontier‑Daten als CSV", data=frontier_csv, file_name="efficient_frontier.csv", mime="text/csv")

# =========================================
# Tests & Sanity Checks
# =========================================

with st.expander("🔎 Built‑in Tests & Sanity Checks"):
    results = []
    results.append(("Sum(w)=1", np.isclose(w_opt.sum(), 1.0, atol=1e-6)))
    results.append(("Long‑only respektiert", (w_opt.values >= -1e-8).all() if long_only else True))
    try:
        np.linalg.cholesky(cov.values + 1e-12 * np.eye(cov.shape[0]))
        psd_ok = True
    except np.linalg.LinAlgError:
        psd_ok = False
    results.append(("Σ ist (nahezu) PSD", psd_ok))
    results.append(("Frontier‑Punkte > 5", len(ef_rets) > 5))
    results.append(("Keine NaNs in Gewichten", not np.isnan(w_opt.values).any()))
    results.append(("μ im plausiblen Bereich (−50%..+100%)", (mu.between(-0.5, 1.0)).all()))

    for name, ok in results:
        st.write(("✅" if ok else "❌"), name)

# =========================================
# Hinweise
# =========================================
with st.expander("Methodik & Caveats"):
    st.markdown(
        """
        **Methodik**
        - Log‑Returns aus Preisen (oder `log(1+r)` bei arithm. Returns); Annualisierung: Daily=252, Weekly=52, Monthly=12.
        - Optimierung über Projected‑Gradient (PGD):
          - GMV: Minimierung w'Σw; Ziel‑Rendite via Penalty (w·μ ≥ target)
          - Tangency: Maximierung Sharpe (μ−rf)/σ
          - Constraints: Sum(w)=1, 0≤w≤Cap (Long‑only)
        - Frontier: Sweep über Zielrenditen zwischen min(μ) und max(μ).

        **Caveats**
        - CSV‑Interpretation sorgfältig wählen (Preise vs. Renditen, Prozent vs. Dezimal, Log vs. arithm.).
        - Historische μ ≠ Zukunft; für Thesis/Investor‑Pitch evtl. Black‑Litterman/Empirical‑Bayes nutzen.
        - Keine Transaktionskosten, Steuern, Liquidität, Rebalancing berücksichtigt.
        """
    )
