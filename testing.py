import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from math import log, sqrt, exp

# ---------- Black-Scholes and Greeks ----------
def black_scholes(S, K, T, r, sigma, opt_type='call'):
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if opt_type == 'call':
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def greeks(S, K, T, r, sigma, opt_type='call'):
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    delta = norm.cdf(d1) if opt_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    vega = S * norm.pdf(d1) * sqrt(T) / 100
    theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) -
             r * K * exp(-r * T) * (norm.cdf(d2) if opt_type == 'call' else norm.cdf(-d2))) / 365
    rho = (K * T * exp(-r * T) * (norm.cdf(d2) if opt_type == 'call' else -norm.cdf(-d2))) / 100
    return delta, gamma, vega, theta, rho

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")
st.title("📊 Black-Scholes Option Pricing & Analytics Dashboard")

# ---------- Sidebar Inputs ----------
st.sidebar.header("🧾 Input Parameters")

S = st.sidebar.slider("Underlying Price (S)", 10.0, 500.0, 100.0)
K = st.sidebar.slider("Strike Price (K)", 10.0, 500.0, 100.0)
T = st.sidebar.slider("Time to Maturity (T in Years)", 0.01, 2.0, 1.0)
r = st.sidebar.slider("Risk-Free Rate (r)", 0.00, 0.2, 0.05)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.2)
opt_type = st.sidebar.radio("Option Type", ['call', 'put'])
premium = st.sidebar.number_input("Option Premium Paid", min_value=0.0, value=10.0, step=0.5)

# Heatmap Ranges
st.sidebar.markdown("### 📈 Heatmap Ranges")
strike_range = st.sidebar.slider("Strike Range", 50, 150, (80, 120))
vol_range = st.sidebar.slider("Volatility Range", 0.1, 1.0, (0.1, 0.5))

# ---------- Price & Greeks ----------
st.header("🧮 Option Price and Greeks")

price = black_scholes(S, K, T, r, sigma, opt_type)
delta, gamma, vega, theta, rho = greeks(S, K, T, r, sigma, opt_type)

col1, col2, col3 = st.columns(3)
col1.metric("Option Price", f"${price:.2f}")
col2.metric("Delta", f"{delta:.4f}")
col3.metric("Gamma", f"{gamma:.4f}")

col4, col5 = st.columns(2)
col4.metric("Vega", f"{vega:.4f}")
col5.metric("Theta", f"{theta:.4f}")

st.metric("Rho", f"{rho:.4f}")

# ---------- Price vs Volatility Plot ----------
st.markdown("### 📉 Option Price vs Volatility")
vol_array = np.linspace(0.01, 1.0, 100)
price_array = [black_scholes(S, K, T, r, v, opt_type) for v in vol_array]

fig, ax = plt.subplots()
ax.plot(vol_array, price_array, color='blue')
ax.set_xlabel("Volatility (σ)")
ax.set_ylabel("Option Price")
ax.set_title("Option Price vs Volatility")
st.pyplot(fig)

# ---------- Heatmap Tab ----------
st.markdown("### 🔥 Heatmaps: Option Price and Delta")
strike_vals = np.arange(strike_range[0], strike_range[1] + 1, 2)
vol_vals = np.linspace(vol_range[0], vol_range[1], 30)

price_matrix = np.zeros((len(vol_vals), len(strike_vals)))
delta_matrix = np.zeros_like(price_matrix)
pl_matrix = np.zeros_like(price_matrix)

for i, vol in enumerate(vol_vals):
    for j, strike in enumerate(strike_vals):
        p = black_scholes(S, strike, T, r, vol, opt_type)
        d = greeks(S, strike, T, r, vol, opt_type)[0]
        price_matrix[i, j] = p
        delta_matrix[i, j] = d
        pl_matrix[i, j] = p - premium

# --- Price Heatmap ---
st.markdown("#### 🎯 Option Price Heatmap")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.heatmap(price_matrix, xticklabels=strike_vals, yticklabels=np.round(vol_vals, 2),
            cmap="YlGnBu", ax=ax1, cbar_kws={'label': 'Price'})
ax1.set_xlabel("Strike Price (K)")
ax1.set_ylabel("Volatility (σ)")
st.pyplot(fig1)

# --- Delta Heatmap ---
st.markdown("#### 📘 Delta Heatmap")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.heatmap(delta_matrix, xticklabels=strike_vals, yticklabels=np.round(vol_vals, 2),
            cmap="coolwarm", center=0, ax=ax2, cbar_kws={'label': 'Delta'})
ax2.set_xlabel("Strike Price (K)")
ax2.set_ylabel("Volatility (σ)")
st.pyplot(fig2)

# --- P&L Heatmap ---
st.markdown("#### 💹 P&L Heatmap (Relative to Premium Paid)")
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.heatmap(pl_matrix, xticklabels=strike_vals, yticklabels=np.round(vol_vals, 2),
            cmap="RdYlGn", center=0, ax=ax3, cbar_kws={'label': 'Profit/Loss'})
ax3.set_xlabel("Strike Price (K)")
ax3.set_ylabel("Volatility (σ)")
st.pyplot(fig3)
