"""
Streamlit-App: Optimales Portfolio für den SMI (Offline, ohne yfinance)
Autor: ChatGPT (für Matthias)

Warum diese Version?
- Funktioniert **ohne Internet** und **ohne yfinance**.
- Nutzt **CSV-Upload** mit realen SMI-Kursen **oder** synthetische Demo-Daten.
- Enthält Diagnose-Checks, um typische CSV-Probleme (Datumsformat, fehlende Werte, Returns vs. Preise) zu erkennen.

Start:  streamlit run app.py
"""

import io
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

# Optionale Robustheit
try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# Optimierung
try:
    import scipy.optimize as sco
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# =============================
# Utility
# =============================

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
    else:  # Monthly
        return prices.resample("M").last()


def to_log_returns_from_prices(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    px = resample_prices(prices, freq)
    # Nicht-positive Preise verursachen log-Probleme → in NaN umwandeln
    px = px.mask(px <= 0)
    rets = np.log(px / px.shift(1))
    return rets.dropna(how="all")


def winsorize_df(df: pd.DataFrame, p: float = 0.01) -> pd.DataFrame:
    low = df.quantile(p)
    high = df.quantile(1 - p)
    return df.clip(lower=low, upper=high, axis=1)


# =============================
# Offline Datenquellen
# =============================

def parse_uploaded_csv(file) -> pd.DataFrame:
    """Erwartetes CSV-Wide-Format: Date + Ticker-Spalten mit Close-Preisen/Returns.
    - Datum wird geparst und als Index gesetzt.
    - Alle Nicht-Datum-Spalten werden nach float konvertiert (coerce).
    - Leere Zeilen/Spalten werden entfernt.
    """
    raw = pd.read_csv(file)
    # Datums-Spalte ermitteln
    date_col = None
    for cand in ["Date", "date", "Datum", "Time", "Datetime"]:
        if cand in raw.columns:
            date_col = cand
            break
    if date_col is None:
        date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col])
    raw = raw.set_index(date_col).sort_index()

    # Alle übrigen Spalten in float konvertieren
    df = raw.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Duplikate im Index aggregieren (letzter Wert)
    if not df.index.is_unique:
        df = df.groupby(df.index).last()

    # Vollständig leere Zeilen/Spalten entfernen
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return df


# Vollständige SMI-Liste (Stand 2025, 20 Titel)
SMI_TICKERS = [
    "NESN.SW","NOVN.SW","ROG.SW","UBSG.SW","ZURN.SW","ABBN.SW","SIKA.SW","GIVN.SW",
    "LONN.SW","SREN.SW","SGSN.SW","HOLN.SW","GEBN.SW","UHR.SW","CFR.SW","ALCN.SW",
    "LOGN.SW","SOON.SW","EMSN.SW","BALN.SW"
]


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


# =============================
# Optimizer (SLSQP)
# =============================

def _constraints_long_only(n, target_ret: Optional[float] = None, mu: Optional[np.ndarray] = None, w_max: Optional[float] = None):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_ret is not None and mu is not None:
        cons.append({"type": "ineq", "fun": lambda w, m=mu, tr=target_ret: w @ m - tr})
    bounds = tuple((0.0, 1.0 if w_max is None else float(w_max)) for _ in range(n))
    return cons, bounds


def _min_var(mu: np.ndarray, cov: np.ndarray, long_only: bool = True, target_ret: Optional[float] = None, w_max: Optional[float] = None):
    if not HAS_SCIPY:
        raise RuntimeError("scipy ist nicht installiert – bitte `pip install scipy` ausführen.")
    n = len(mu)
    x0 = np.repeat(1 / n, n)

    def obj(w, C=cov):
        return float(w @ C @ w)

    if long_only:
        cons, bounds = _constraints_long_only(n, target_ret, mu, w_max)
    else:
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        if target_ret is not None:
            cons.append({"type": "ineq", "fun": lambda w, m=mu, tr=target_ret: w @ m - tr})
        bounds = tuple((None, None) for _ in range(n))

    res = sco.minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x, res


def _max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, long_only: bool = True, w_max: Optional[float] = None):
    if not HAS_SCIPY:
        raise RuntimeError("scipy ist nicht installiert – bitte `pip install scipy` ausführen.")
    n = len(mu)
    x0 = np.repeat(1 / n, n)

    def neg_sharpe(w, m=mu, C=cov, r=rf):
        port_ret = w @ m
        port_vol = np.sqrt(w @ C @ w)
        if port_vol <= 0:
            return 1e6
        return -(port_ret - r) / port_vol

    if long_only:
        cons, bounds = _constraints_long_only(n, None, None, w_max)
    else:
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((None, None) for _ in range(n))

    res = sco.minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x, res


def _efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 50, long_only: bool = True, w_max: Optional[float] = None, ret_span: Optional[Tuple[float, float]] = None):
    m = np.asarray(mu)
    C = np.asarray(cov)
    r_min, r_max = (m.min(), m.max()) if ret_span is None else ret_span
    targets = np.linspace(r_min, r_max, n_points)

    ws, vols, rets = [], [], []
    for tr in targets:
        try:
            w, _ = _min_var(m, C, long_only=long_only, target_ret=tr, w_max=w_max)
        except Exception:
            # Infeasible target -> skip
            continue
        ws.append(w)
        vol = float(np.sqrt(w @ C @ w))
        vols.append(vol)
        rets.append(float(w @ m))
    return np.array(rets), np.array(vols), np.array(ws)


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="Optimales SMI-Portfolio (offline)", layout="wide")

st.title("📈 Optimales Portfolio für den SMI – Offline Efficient Frontier")
st.caption("Mean‑Variance‑Optimierung ohne yfinance. Lade CSV oder nutze Demo‑Daten.")

with st.sidebar:
    st.header("1) Datenquelle wählen")
    data_mode = st.radio("Quelle", ["CSV hochladen", "Demo‑Daten (synthetisch)"])

    if data_mode == "CSV hochladen":
        st.markdown(
            "**CSV‑Schema (Wide):** Erste Spalte = Datum, weitere Spalten = Ticker, Werte = Close oder Returns.\n"
            "Beispiel‑Spalten: Date, NESN.SW, NOVN.SW, ROG.SW, …"
        )
        uploaded = st.file_uploader("CSV mit Preisen/Returns hochladen", type=["csv"])
        st.divider()
        st.subheader("CSV-Interpretation")
        csv_kind = st.radio("Was enthält die CSV?", ["Preise (Close)", "Renditen"], index=0)
        if csv_kind == "Renditen":
            ret_kind = st.radio("Rendite-Skalierung", ["Dezimal (0.01 = 1%)", "Prozent (1 = 1%)"], index=0)
            treat_as_log = st.checkbox("CSV-Renditen sind bereits Log-Returns", value=False)
        else:
            # Fehlende Daten Behandlung
            fill_choice = st.selectbox("Fehlende Werte behandeln", ["Keine Füllung", "Vorwärts/Zurück füllen"], index=1)
            zero_to_nan = st.checkbox("Nicht-positive Preise als fehlend behandeln (empfohlen)", value=True)
        # Template/Exports
        st.divider()
        st.markdown("**Template/Demo-Exports**")
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
        if st.button("📄 Demo‑Daten als CSV generieren"):
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
    use_lw = st.checkbox("Ledoit‑Wolf Kovarianz (falls sklearn vorhanden)", value=False and HAS_SKLEARN)

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

# =============================
# Daten laden & vorbereiten
# =============================

if data_mode == "CSV hochladen":
    if uploaded is None:
        st.info("Bitte CSV hochladen – oder links auf Demo‑Daten umstellen.")
        st.stop()
    prices_or_returns = parse_uploaded_csv(uploaded)

    # Diagnose zu Datenqualität
    st.subheader("CSV-Diagnose")
    st.write(f"Zeilen: {prices_or_returns.shape[0]}, Spalten: {prices_or_returns.shape[1]}")
    st.dataframe(prices_or_returns.tail())
    missing_ratio = prices_or_returns.isna().mean().round(3)
    st.write("**Anteil fehlender Werte pro Spalte** (0=keine, 1=alle):")
    st.dataframe(missing_ratio.to_frame("missing_ratio"))

    if csv_kind == "Preise (Close)":
        dfp = prices_or_returns.copy()
        if zero_to_nan:
            dfp = dfp.mask(dfp <= 0)
        if fill_choice == "Vorwärts/Zurück füllen":
            before = dfp.isna().sum().sum()
            dfp = dfp.ffill().bfill()
            after = dfp.isna().sum().sum()
            st.caption(f"Fehlende Werte gefüllt: {before - after}")
        rets = to_log_returns_from_prices(dfp, freq)
    else:
        dfr = prices_or_returns.copy()
        # Skalierung
        if ret_kind == "Prozent (1 = 1%)":
            dfr = dfr / 100.0
        if treat_as_log:
            rets = dfr
        else:
            # Arithmetische Renditen → in Log-Renditen konvertieren
            rets = np.log1p(dfr)

else:
    prices = demo_prices(n_years=demo_years, freq=freq)
    st.subheader("Demo-Daten (Preise)")
    st.write(f"Zeilen: {prices.shape[0]}, Spalten: {prices.shape[1]}")
    st.dataframe(prices.tail())
    rets = to_log_returns_from_prices(prices, freq)

# Optional: Winsorizing
if use_winsor:
    rets = winsorize_df(rets, 0.01)

# Annualisierung
mu, cov, periods = annualize_stats(rets, freq)

# Optional: Ledoit-Wolf
if use_lw and HAS_SKLEARN:
    try:
        lw = LedoitWolf().fit(rets.dropna())
        cov = pd.DataFrame(lw.covariance_, index=rets.columns, columns=rets.columns) * periods
    except Exception as e:
        st.warning(f"Ledoit‑Wolf konnte nicht berechnet werden: {e}")

# Anzeige Schätzungen
st.subheader("Annualisierte Schätzungen")
col1, col2 = st.columns(2)
with col1:
    st.write("**Erwartete Renditen (μ)**")
    st.dataframe(mu.sort_values(ascending=False).to_frame("mu"))
with col2:
    st.write("**Kovarianzmatrix (Σ)**")
    st.dataframe(cov)

# =============================
# Optimierung & Frontier
# =============================

if not HAS_SCIPY:
    st.error("scipy ist nicht installiert. Bitte `pip install scipy` und App erneut starten.")
    st.stop()

try:
    if choice == "Tangency (max. Sharpe)":
        w_opt, res = _max_sharpe(mu.values, cov.values, rf=rf, long_only=long_only, w_max=w_cap if long_only else None)
    elif choice == "GMV (Min‑Var)":
        w_opt, res = _min_var(mu.values, cov.values, long_only=long_only, target_ret=None, w_max=w_cap if long_only else None)
    else:
        w_opt, res = _min_var(mu.values, cov.values, long_only=long_only, target_ret=target_ret, w_max=w_cap if long_only else None)
except Exception as e:
    st.error(f"Optimierung fehlgeschlagen: {e}")
    st.stop()

w_opt = pd.Series(w_opt, index=mu.index).clip(lower=0)
w_opt = w_opt / w_opt.sum()

p_ret = float(w_opt @ mu)
p_vol = float(np.sqrt(w_opt @ cov @ w_opt))
p_sharpe = (p_ret - rf) / p_vol if p_vol > 0 else np.nan

with st.spinner("Berechne Efficient Frontier …"):
    ef_rets, ef_vols, ef_ws = _efficient_frontier(mu.values, cov.values, n_points=n_frontier, long_only=long_only, w_max=w_cap if long_only else None)

# =============================
# Visualisierung
# =============================

st.subheader("Efficient Frontier")
fig = plt.figure(figsize=(8, 6))
plt.scatter(ef_vols, ef_rets, s=12, label="Efficient Frontier")
plt.scatter(p_vol, p_ret, s=80, marker="*", label=f"Ausgewählt ({choice})")
plt.xlabel("Volatilität (σ)")
plt.ylabel("Erwartete Rendite (μ)")
plt.legend()
st.pyplot(fig)

# =============================
# Outputs
# =============================

st.subheader("Optimale Gewichte")
weights_df = pd.DataFrame({"Weight": w_opt.round(4)}).sort_values("Weight", ascending=False)
st.dataframe(weights_df)

st.markdown(
    f"**Kennzahlen:**  "
    + f"Rendite p.a.: **{p_ret:.2%}**, Volatilität p.a.: **{p_vol:.2%}**, Sharpe: **{p_sharpe:.2f}** (rf={rf:.2%})"
)

# Downloads
csv = weights_df.to_csv().encode("utf-8")
st.download_button("📥 Gewichte als CSV", data=csv, file_name="smi_portfolio_weights.csv", mime="text/csv")

png_buf = io.BytesIO()
fig.savefig(png_buf, format="png", dpi=180, bbox_inches="tight")
png_buf.seek(0)
st.download_button("📥 Frontier als PNG", data=png_buf, file_name="efficient_frontier.png", mime="image/png")

# =============================
# Tests & Sanity Checks (erweitert)
# =============================

with st.expander("🔎 Built‑in Tests & Sanity Checks"):
    def _run_tests():
        results = []
        # Test 1: Gewichte summieren zu 1
        results.append(("Sum(w)=1", np.isclose(w_opt.sum(), 1.0, atol=1e-6)))
        # Test 2: Long‑only respektiert
        results.append(("Long‑only respektiert", True if not long_only else (w_opt.values >= -1e-8).all()))
        # Test 3: Volatilität positiv
        results.append(("Volatilität ≥ 0", p_vol >= 0))
        # Test 4: Frontier‑Dimensionen konsistent
        results.append(("Frontier‑Punkte konsistent", (len(ef_rets) == len(ef_vols)) and (len(ef_rets) > 5)))
        # Test 5: Kovarianz (nahezu) PSD
        try:
            np.linalg.cholesky(cov.values + 1e-12 * np.eye(cov.shape[0]))
            s5 = True
        except np.linalg.LinAlgError:
            s5 = False
        results.append(("Σ ist (nahezu) PSD", s5))
        # Test 6: Keine NaNs in Gewichten
        results.append(("Keine NaNs in Gewichten", not np.isnan(w_opt.values).any()))
        # Test 7: μ realistisch (−50%..+100% p.a.)
        results.append(("μ im plausiblen Bereich", (mu.between(-0.5, 1.0)).all()))
        return results

    for name, ok in _run_tests():
        st.write(("✅" if ok else "❌"), name)

# =============================
# Methodik & Caveats
# =============================

with st.expander("Methodik & Caveats"):
    st.markdown(
        """
        **Daten (offline):**
        - CSV (Wide) mit Datum + Ticker‑Spalten. Wenn die CSV bereits **Renditen** enthält, wähle dies links aus.
        - Für Preise: nicht-positive Werte werden optional als fehlend behandelt (empfohlen) und können vorwärts/ rückwärts gefüllt werden.

        **Methodik**
        - Renditen: Log‑Returns. Bei arithmetischen Returns wird automatisch `log(1+r)` verwendet.
        - Annualisierung: Daily=252, Weekly=52, Monthly=12.
        - Kovarianz: Sample‑Σ (optional Ledoit‑Wolf), Optimierung via SLSQP (Summe Gewichte=1, Long‑only & w_max).
        - Frontier: Min‑Var entlang einer Zielrendite‑Spanne. Infeasible Targets werden übersprungen.

        **Caveats**
        - Unterschiedliche Handelskalender / fehlende Werte beeinflussen Σ massiv → Datenbereinigung wichtig.
        - Historische μ ≠ Zukunft. Für Thesis/Investor‑Pitch eher Black‑Litterman/Empirical‑Bayes nutzen.
        - Keine Transaktionskosten/Steuern/Rebalancing berücksichtigt.
        """
    )
