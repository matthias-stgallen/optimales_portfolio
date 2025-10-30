"""
Streamlit-App: Optimales Portfolio für den SMI (Offline, ohne yfinance)
Autor: ChatGPT (für Matthias)

Warum diese Version?
- Funktioniert **ohne Internet** und **ohne yfinance**.
- Nutzt **CSV-Upload** mit realen SMI-Kursen **oder** synthetische Demo-Daten.
- Entfernt Abhängigkeit von `matplotlib` → nutzt stattdessen `streamlit`/`plotly` für Visualisierung.

Start:  streamlit run app.py
"""

import io
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date

# =============================
# Utility: Projektion auf gekappte Simplex {w: sum(w)=1, 0<=w<=cap}
# =============================

def _project_capped_simplex(y: np.ndarray, cap: float = 1.0, s: float = 1.0) -> np.ndarray:
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


def _normalize_sum1(w: np.ndarray) -> np.ndarray:
    s = w.sum()
    if s == 0:
        return np.full_like(w, 1.0 / len(w))
    return w / s

# =============================
# Optimizer (Projected Gradient)
# =============================

def _pg_min_var(mu: np.ndarray, cov: np.ndarray, target_ret: Optional[float], long_only: bool, w_cap: Optional[float], max_iter: int = 2000) -> np.ndarray:
    n = len(mu)
    w = np.full(n, 1.0 / n)
    C = cov
    cap = 1.0 if (w_cap is None) else float(w_cap)
    beta = 1000.0 if target_ret is not None else 0.0
    tr = target_ret if target_ret is not None else -1e9
    t = 0.1
    eps = 1e-9
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
        if np.linalg.norm(grad, ord=2) < 1e-6:
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
        grad = -(mu) / b + (a * Cw) / (b ** 3)
        y = w - t * grad
        if long_only:
            w = _project_capped_simplex(y, cap=cap, s=1.0)
        else:
            w = _normalize_sum1(y)
        t = min(0.2, t * 1.01)
    return _normalize_sum1(np.clip(w, 0, 1)) if long_only else _normalize_sum1(w)


def _min_var(mu: np.ndarray, cov: np.ndarray, long_only: bool = True, target_ret: Optional[float] = None, w_max: Optional[float] = None):
    w = _pg_min_var(mu, cov, target_ret=target_ret, long_only=long_only, w_cap=w_max)
    return w, {"method": "PGD"}


def _max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, long_only: bool = True, w_max: Optional[float] = None):
    w = _pg_max_sharpe(mu, cov, rf=rf, long_only=long_only, w_cap=w_max)
    return w, {"method": "PGD"}

# =============================
# Datenvorbereitung
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
    else:
        return prices.resample("M").last()


def to_log_returns_from_prices(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    px = resample_prices(prices, freq)
    px = px.mask(px <= 0)
    rets = np.log(px / px.shift(1))
    return rets.dropna(how="all")

# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="Optimales SMI-Portfolio (offline)", layout="wide")

st.title("📈 Optimales Portfolio für den SMI – Offline Efficient Frontier")
st.caption("Mean‑Variance‑Optimierung ohne yfinance. Lade CSV oder nutze Demo‑Daten.")

# (gekürzte UI und Datenlogik wie zuvor …)

# =============================
# Visualisierung mit Plotly
# =============================

st.subheader("Efficient Frontier")
fig = go.Figure()
fig.add_trace(go.Scatter(x=ef_vols, y=ef_rets, mode="markers", name="Efficient Frontier"))
fig.add_trace(go.Scatter(x=[p_vol], y=[p_ret], mode="markers", marker=dict(size=12, symbol="star", color="red"), name=f"Ausgewählt ({choice})"))
fig.update_layout(xaxis_title="Volatilität (σ)", yaxis_title="Erwartete Rendite (μ)")
st.plotly_chart(fig, use_container_width=True)

# Downloads CSV bleibt
