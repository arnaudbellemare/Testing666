import ccxt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import t as studentt
from numba import njit
from numpy.typing import NDArray
from typing import Optional
import streamlit as st
import pywt  # for wavelet shrinkage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

###############################################################################
# SESSION STATE & SIDEBAR INPUTS
###############################################################################
if "username" not in st.session_state:
    st.session_state["username"] = "Guest"

st.title("CNO Dashboard")
st.write(f"Welcome, {st.session_state['username']}!")

lookback_options = {
    "1 Day": 1440,
    "3 Days": 4320,
    "1 Week": 10080,
    "2 Weeks": 20160,
    "1 Month": 43200
}
global_lookback_label = st.sidebar.selectbox(
    "Select Global Lookback Period",
    list(lookback_options.keys()),
    key="global_lookback_label"
)
global_lookback_minutes = lookback_options[global_lookback_label]
timeframe = st.sidebar.selectbox(
    "Select Timeframe", ["1m", "5m", "15m", "1h"],
    key="timeframe_widget"
)
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["BVC", "ACI"],
    key="analysis_type"
)
st.write("Current analysis type:", analysis_type)

# Ticker selection for main indicator analysis (BVC/ACI)
ticker_main = st.sidebar.text_input("Enter Ticker Symbol for BVC/ACI Analysis", value="BTC/USD", key="ticker_main")

###############################################################################
# 1) FETCH DATA FUNCTION
###############################################################################
def fetch_data(symbol="BTC/USD", timeframe="1m", lookback_minutes=1440):
    exchange = ccxt.kraken()
    now_ms = exchange.milliseconds()
    cutoff_ts = now_ms - lookback_minutes * 60 * 1000
    all_ohlcv = []
    since = cutoff_ts
    max_limit = 1440
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=max_limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        last_timestamp = ohlcv[-1][0]
        if last_timestamp <= cutoff_ts or len(ohlcv) < max_limit:
            break
        since = last_timestamp + 1
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["stamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.sort_values("stamp").reset_index(drop=True)

###############################################################################
# 2) HELPER FUNCTIONS
###############################################################################
@njit(cache=True)
def ema(arr_in: NDArray, window: int, alpha: Optional[float] = 0) -> NDArray:
    alpha = 3 / float(window + 1) if alpha == 0 else alpha
    n = arr_in.size
    ewma = np.empty(n, dtype=np.float64)
    ewma[0] = arr_in[0]
    for i in range(1, n):
        ewma[i] = (arr_in[i] * alpha) + (ewma[i-1] * (1 - alpha))
    return ewma

def gradual_normalize(values: np.ndarray, window: int = 50, scale: float = 1e4) -> np.ndarray:
    n = len(values)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        window_vals = values[start_idx : i + 1]
        max_abs_val = np.max(np.abs(window_vals))
        if max_abs_val == 0:
            out[i] = 0
        else:
            out[i] = values[i] / max_abs_val * scale
    return out

def compute_investment_performance(data, labels):
    returns = np.diff(data) / data[:-1]
    strat_returns = [returns[i] if labels[i] == 1 else -returns[i] for i in range(len(returns))]
    return np.prod(1 + np.array(strat_returns)) - 1

def wavelet_shrinkage(data, wavelet='db4', level=2):
    coeff = pywt.wavedec(data, wavelet, mode='per')
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff_thresh = [coeff[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeff[1:]]
    data_denoised = pywt.waverec(coeff_thresh, wavelet, mode='per')
    return data_denoised[:len(data)]

###############################################################################
# 3) INDICATOR CLASSES (BVC and ACIBVC)
###############################################################################
class HawkesBVC:
    def __init__(self, window=20, kappa=0.1, dof=0.25):
        self.window = window
        self.kappa = kappa
        self.dof = dof

    def _label(self, r, sigma):
        if sigma > 0.0:
            return 2 * studentt.cdf(r / sigma, df=self.dof) - 1.0
        else:
            return 0.0

    def eval(self, df: pd.DataFrame, scale=1e4):
        df = df.copy().sort_values("stamp")
        prices = df["close"]
        cumr = np.log(prices / prices.iloc[0])
        r = cumr.diff().fillna(0.0)
        # Assumes a "volume" column exists
        volume = df["volume"]
        sigma = r.rolling(self.window).std().fillna(0.0)
        alpha_exp = np.exp(-self.kappa)
        labels = np.array([self._label(r.iloc[i], sigma.iloc[i]) for i in range(len(r))])
        bvc = np.zeros(len(volume), dtype=float)
        current_bvc = 0.0
        for i in range(len(volume)):
            current_bvc = current_bvc * alpha_exp + volume.values[i] * labels[i]
            bvc[i] = current_bvc
        max_abs = np.max(np.abs(bvc))
        if max_abs != 0:
            bvc = bvc / max_abs * scale
        return pd.DataFrame({"stamp": df["stamp"], "bvc": bvc})

class ACIBVC:
    def __init__(self, kappa=0.1):
        self.kappa = kappa

    def estimate_intensity(self, times, beta):
        intensities = [0.0]
        for i in range(1, len(times)):
            delta_t = times[i] - times[i-1]
            intensities.append(intensities[-1] * np.exp(-beta * delta_t) + 1)
        return np.array(intensities)

    def eval(self, df: pd.DataFrame, scale=1e5):
        df = df.copy().sort_values("stamp")
        df["time_s"] = df["stamp"].astype(np.int64) // 10**9
        times = df["time_s"].values
        intensities = self.estimate_intensity(times, self.kappa)
        df = df.iloc[:len(intensities)]
        df["intensity"] = intensities
        df["price_change"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        df["label"] = df["intensity"] * df["price_change"]
        df["weighted_volume"] = df["volume"] * df["label"]
        alpha_exp = np.exp(-self.kappa)
        bvc_list = []
        current_bvc = 0.0
        for wv in df["weighted_volume"].values:
            current_bvc = current_bvc * alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc = np.array(bvc_list)
        max_abs = np.max(np.abs(bvc))
        if max_abs != 0:
            bvc = bvc / max_abs * scale
        df["bvc"] = bvc
        return df[["stamp", "bvc"]].copy()

###############################################################################
# 4) TUNING FUNCTIONS
###############################################################################
def tune_kappa_classification(df_prices, kappa_grid=None, scale=1e4, indicator_type='hawkes'):
    if kappa_grid is None:
        kappa_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    best_kappa = None
    best_f1 = -np.inf
    best_metrics = None
    df_temp = df_prices.copy().sort_values("stamp")
    close_vals = df_temp["close"].values
    gt_labels = np.zeros(len(close_vals))
    for i in range(len(close_vals) - 1):
        gt_labels[i] = 1 if close_vals[i+1] > close_vals[i] else -1
    gt_labels[-1] = gt_labels[-2]
    
    for k in kappa_grid:
        if indicator_type == 'hawkes':
            model = HawkesBVC(window=20, kappa=k)
        else:
            model = ACIBVC(kappa=k)
        indicator_df = model.eval(df_temp.copy(), scale=scale)
        merged = df_temp.merge(indicator_df, on="stamp", how="inner")
        pred_labels = np.where(merged["bvc"].values >= 0, 1, -1)
        accuracy = accuracy_score(gt_labels[:len(pred_labels)], pred_labels)
        precision = precision_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        recall = recall_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        f1 = f1_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        gt_bin = (gt_labels[:len(pred_labels)] == 1).astype(int)
        pred_bin = (pred_labels == 1).astype(int)
        try:
            auc = roc_auc_score(gt_bin, pred_bin)
        except Exception:
            auc = 0.5
        net_yield = compute_investment_performance(merged["close"].values, pred_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_kappa = k
            best_metrics = (accuracy, precision, recall, f1, auc, net_yield)
    return best_kappa, best_metrics

def tune_window_classification(df_prices, window_grid=None, kappa=1e-4, scale=1e4, indicator_type='hawkes'):
    if window_grid is None:
        window_grid = [10, 15, 20, 25, 30]
    best_window = None
    best_f1 = -np.inf
    best_metrics = None
    df_temp = df_prices.copy().sort_values("stamp")
    close_vals = df_temp["close"].values
    gt_labels = np.zeros(len(close_vals))
    for i in range(len(close_vals) - 1):
        gt_labels[i] = 1 if close_vals[i+1] > close_vals[i] else -1
    gt_labels[-1] = gt_labels[-2]
    
    for w in window_grid:
        if indicator_type == 'hawkes':
            model = HawkesBVC(window=w, kappa=kappa)
        else:
            continue
        indicator_df = model.eval(df_temp.copy(), scale=scale)
        merged = df_temp.merge(indicator_df, on="stamp", how="inner")
        pred_labels = np.where(merged["bvc"].values >= 0, 1, -1)
        accuracy = accuracy_score(gt_labels[:len(pred_labels)], pred_labels)
        precision = precision_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        recall = recall_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        f1 = f1_score(gt_labels[:len(pred_labels)], pred_labels, pos_label=1)
        gt_bin = (gt_labels[:len(pred_labels)] == 1).astype(int)
        pred_bin = (pred_labels == 1).astype(int)
        try:
            auc = roc_auc_score(gt_bin, pred_bin)
        except Exception:
            auc = 0.5
        net_yield = compute_investment_performance(merged["close"].values, pred_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_window = w
            best_metrics = (accuracy, precision, recall, f1, auc, net_yield)
    return best_window, best_metrics

###############################################################################
# 5) AUTO LABELING FUNCTION (for momentum signal using wavelet-denoised prices)
###############################################################################
def auto_labeling(data_list, timestamp_list, w):
    labels = np.zeros(len(data_list))
    FP = data_list[0]
    x_H = data_list[0]
    HT = timestamp_list[0]
    x_L = data_list[0]
    LT = timestamp_list[0]
    Cid = 0
    FP_N = 0
    for i in range(len(data_list)):
        if data_list[i] > FP + data_list[0] * w:
            x_H = data_list[i]
            HT = timestamp_list[i]
            FP_N = i
            Cid = 1
            break
        if data_list[i] < FP - data_list[0] * w:
            x_L = data_list[i]
            LT = timestamp_list[i]
            FP_N = i
            Cid = -1
            break
    for i in range(FP_N, len(data_list)):
        if Cid > 0:
            if data_list[i] > x_H:
                x_H = data_list[i]
                HT = timestamp_list[i]
            if data_list[i] < x_H - x_H * w and LT < HT:
                for j in range(len(data_list)):
                    if timestamp_list[j] > LT and timestamp_list[j] <= HT:
                        labels[j] = 1
                x_L = data_list[i]
                LT = timestamp_list[i]
                Cid = -1
        elif Cid < 0:
            if data_list[i] < x_L:
                x_L = data_list[i]
                LT = timestamp_list[i]
            if data_list[i] > x_L + x_L * w and HT <= LT:
                for j in range(len(data_list)):
                    if timestamp_list[j] > HT and timestamp_list[j] <= LT:
                        labels[j] = -1
                x_H = data_list[i]
                HT = timestamp_list[i]
                Cid = 1
    labels[0] = labels[1] if len(labels) > 1 else Cid
    labels = np.where(labels == 0, Cid, labels)
    assert len(labels) == len(timestamp_list)
    timestamp2label_dict = {timestamp_list[i]: labels[i] for i in range(len(timestamp_list))}
    return labels, timestamp2label_dict

###############################################################################
# 6) MAIN SCRIPT (STREAMLIT APP) WITH INDICATOR-NORMALIZED COLORING & MOMENTUM
###############################################################################
st.header("Price & Indicator Analysis")

# Fetch main data
df = fetch_data(symbol=ticker_main, timeframe=timeframe, lookback_minutes=720)
df = df.sort_values("stamp").reset_index(drop=True)
st.write("Data range:", df["stamp"].min(), "to", df["stamp"].max())
st.write("Number of rows:", len(df))

# Compute additional price fields
df["ScaledPrice"] = np.log(df["close"] / df["close"].iloc[0]) * 1e4
df["ScaledPrice_EMA"] = ema(df["ScaledPrice"].values, window=36)

# Compute VWAP
df["cum_vol"] = df["volume"].cumsum()
df["cum_pv"] = (df["close"] * df["volume"]).cumsum()
df["vwap"] = df["cum_pv"] / df["cum_vol"]
if df["vwap"].iloc[0] == 0 or not np.isfinite(df["vwap"].iloc[0]):
    df["vwap_transformed"] = df["ScaledPrice"]
else:
    df["vwap_transformed"] = np.log(df["vwap"] / df["vwap"].iloc[0]) * 1e4

# --- Compute Indicator based on Analysis Type using optimal kappa (via classification metrics)
if analysis_type == "BVC":
    st.write("### BVC Analysis (Optimized via Classification Metrics)")
    optimal_kappa, metrics = tune_kappa_classification(df, kappa_grid=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
                                                     scale=1e4, indicator_type='hawkes')
    st.write(f"Optimal kappa: {optimal_kappa}")
    st.write(f"Accuracy: {metrics[0]:.4f}")
    st.write(f"Precision: {metrics[1]:.4f}")
    st.write(f"Recall: {metrics[2]:.4f}")
    st.write(f"F1 Score: {metrics[3]:.4f}")
    st.write(f"AUC: {metrics[4]:.4f}")
    st.write(f"Net Yield: {metrics[5]:.4f}")
    optimal_window, window_metrics = tune_window_classification(df, window_grid=[10, 15, 20, 25, 30],
                                                                kappa=optimal_kappa, scale=1e4, indicator_type='hawkes')
    st.write(f"Optimal window: {optimal_window}")
    st.write(f"Window Metrics - Accuracy: {window_metrics[0]:.4f}, Precision: {window_metrics[1]:.4f}, Recall: {window_metrics[2]:.4f}, F1 Score: {window_metrics[3]:.4f}, AUC: {window_metrics[4]:.4f}, Net Yield: {window_metrics[5]:.4f}")
    indicator_title = "BVC"
    indicator_df = HawkesBVC(window=optimal_window, kappa=optimal_kappa).eval(df.copy(), scale=1e4)
    
elif analysis_type == "ACI":
    st.write("### ACIBVC Analysis (Optimized via Classification Metrics)")
    optimal_kappa, metrics = tune_kappa_classification(df, kappa_grid=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
                                                     scale=1e5, indicator_type='aci')
    st.write(f"Optimal kappa: {optimal_kappa}")
    st.write(f"Accuracy: {metrics[0]:.4f}")
    st.write(f"Precision: {metrics[1]:.4f}")
    st.write(f"Recall: {metrics[2]:.4f}")
    st.write(f"F1 Score: {metrics[3]:.4f}")
    st.write(f"AUC: {metrics[4]:.4f}")
    st.write(f"Net Yield: {metrics[5]:.4f}")
    indicator_title = "ACI"
    indicator_df = ACIBVC(kappa=optimal_kappa).eval(df.copy(), scale=1e5)

# Merge the indicator into the main DataFrame
df_merged = df.merge(indicator_df, on="stamp", how="inner")
df_merged = df_merged.sort_values("stamp")
df_merged["bvc"] = df_merged["bvc"].fillna(method="ffill").fillna(0)

# --- Compute Momentum Signal using wavelet-denoised auto labeling on the close prices
denoised_close = wavelet_shrinkage(df["close"].values.astype(np.float64), wavelet='db4', level=2)
momentum_labels, _ = auto_labeling(denoised_close, df["stamp"].values, w=0.0003)
df_merged["momentum"] = momentum_labels
# Create momentum-adjusted indicator signal
df_merged["indicator_momentum"] = df_merged["bvc"] * df_merged["momentum"]

###############################################################################
# PLOTTING: Main Price Chart Colored by Momentum-Adjusted Indicator
###############################################################################
fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
# Normalize the momentum-adjusted indicator for coloring
norm_indicator = plt.Normalize(df_merged["indicator_momentum"].min(), df_merged["indicator_momentum"].max())

for i in range(len(df_merged) - 1):
    xvals = df_merged["stamp"].iloc[i:i+2]
    yvals = df_merged["ScaledPrice"].iloc[i:i+2]
    indicator_mom_val = df_merged["indicator_momentum"].iloc[i]
    # Use Blues if positive, Reds if negative
    cmap = plt.cm.Blues if indicator_mom_val >= 0 else plt.cm.Reds
    base_color = cmap(norm_indicator(indicator_mom_val))
    # Darken the color slightly (multiply RGB channels by 0.8)
    darker_color = (0.8 * base_color[0], 0.8 * base_color[1], 0.8 * base_color[2], base_color[3])
    ax.plot(xvals, yvals, color=darker_color, linewidth=1)

# Overlay EMA line
ax.plot(df_merged["stamp"], df_merged["ScaledPrice_EMA"], color="black", linewidth=1, label="EMA(10)")

# Overlay VWAP line (conditional coloring)
for i in range(len(df_merged) - 1):
    xvals = df_merged["stamp"].iloc[i:i+2]
    yvals = df_merged["vwap_transformed"].iloc[i:i+2]
    vwap_color = "blue" if df_merged["ScaledPrice"].iloc[i] > df_merged["vwap_transformed"].iloc[i] else "red"
    ax.plot(xvals, yvals, color=vwap_color, linewidth=1)

# Add watermark text: ticker (in light gray, smaller) near the top center and "CNO" below it
ax.text(0.5, 0.55, ticker_main, transform=ax.transAxes, fontsize=16, color="lightgray",
        alpha=0.3, ha="center", va="center", zorder=0)
ax.text(0.5, 0.45, "CNO", transform=ax.transAxes, fontsize=12, color="lightgray",
        alpha=0.3, ha="center", va="center", zorder=0)

ax.set_xlabel("Time", fontsize=8)
ax.set_ylabel("Scaled Price", fontsize=8)
ax.set_title(f"Price with EMA & VWAP (Colored by {indicator_title} Momentum)", fontsize=10)
ax.legend(fontsize=7)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
plt.setp(ax.get_yticklabels(), fontsize=7)
ax.set_ylim(df_merged["ScaledPrice"].min() - 50, df_merged["ScaledPrice"].max() + 50)
plt.tight_layout()
st.pyplot(fig)

###############################################################################
# PLOTTING: Momentum-Adjusted Indicator Signal
###############################################################################
fig_mom, ax_mom = plt.subplots(figsize=(10, 3), dpi=120)
ax_mom.plot(df_merged["stamp"], df_merged["indicator_momentum"], color="green", linewidth=1, label=f"{indicator_title} with Momentum")
ax_mom.set_xlabel("Time", fontsize=8)
ax_mom.set_ylabel(f"{indicator_title} (Momentum Adjusted)", fontsize=8)
ax_mom.legend(fontsize=7)
ax_mom.set_title(f"{indicator_title} Indicator Adjusted for Momentum", fontsize=10)
ax_mom.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_mom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
plt.setp(ax_mom.get_xticklabels(), rotation=30, ha="right", fontsize=7)
plt.setp(ax_mom.get_yticklabels(), fontsize=7)
plt.tight_layout()
st.pyplot(fig_mom)

###############################################################################
# SECTION 5: Auto Labeling with Ticker Data & EMA (for reference)
###############################################################################
st.header("Section 5: Auto Labeling with Ticker Data & EMA")
ticker_al = st.sidebar.text_input("Enter Ticker Symbol for Auto Labeling", value="BTC/USD", key="ticker_al_6")
lookback_al = global_lookback_minutes
ema_window = st.sidebar.number_input("EMA Window for Chart", min_value=1, max_value=100, value=10, step=1, key="ema_window_6")
try:
    df_al = fetch_data(symbol=ticker_al, timeframe=timeframe, lookback_minutes=lookback_al)
except Exception as e:
    st.error(f"Error fetching ticker data: {e}")
    st.stop()
df_al = df_al.dropna(subset=["close"])
df_al["stamp"] = pd.to_datetime(df_al["timestamp"], unit="ms")
data_al = df_al["close"].values.astype(np.float64)
timestamps_al = df_al["stamp"].values
ema_al = ema(data_al, int(ema_window))
labels_al, timestamp2label_dict_al = auto_labeling(data_al, timestamps_al, w=0.0003)
fig_al, ax_al = plt.subplots(figsize=(10, 4))
point_colors = ['#10a4f4' if label == 1 else 'red' for label in labels_al]
ax_al.scatter(timestamps_al, data_al, c=point_colors, s=5, label="Data Points", zorder=5)
ax_al.plot(timestamps_al, data_al, color="gray", linewidth=0.8, alpha=0.7, label="Close Price")
ax_al.plot(timestamps_al, ema_al, color="black", linewidth=0.5, label=f"EMA({ema_window})")
ax_al.set_title(f"Auto Labeling and EMA for {ticker_al}")
ax_al.set_xlabel("Timestamp")
ax_al.set_ylabel("Price")
plt.xticks(rotation=30)
ax_al.legend()
st.pyplot(fig_al)

###############################################################################
# SECTION 6: Auto Labeling Using Volatility-Influenced Threshold (for reference)
###############################################################################
st.header("Section 6: Auto Labeling Using Volatility-Influenced Threshold")
ticker_al = st.sidebar.text_input("Enter Ticker Symbol for Auto Labeling", value="BTC/USDT", key="ticker_al_7")
lookback_al = global_lookback_minutes
ema_window = st.sidebar.number_input("EMA Window for Chart", min_value=1, max_value=100, value=10, step=1, key="ema_window_7")
try:
    df_al = fetch_data(symbol=ticker_al, timeframe=timeframe, lookback_minutes=lookback_al)
except Exception as e:
    st.error(f"Error fetching ticker data for auto labeling: {e}")
    st.stop()
df_al = df_al.dropna(subset=["close"])
df_al["stamp"] = pd.to_datetime(df_al["timestamp"], unit="ms")
train_frac = 0.8
split_idx = int(len(df_al) * train_frac)
df_train = df_al.iloc[:split_idx].reset_index(drop=True)
data_train = df_train["close"].values.astype(np.float64)
timestamps_train = df_train["stamp"].values
ema_train = ema(data_train, int(ema_window))
def wavelet_shrinkage(data, wavelet='db4', level=2):
    coeff = pywt.wavedec(data, wavelet, mode='per')
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff_thresh = [coeff[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeff[1:]]
    data_denoised = pywt.waverec(coeff_thresh, wavelet, mode='per')
    return data_denoised[:len(data)]
data_train_denoised = wavelet_shrinkage(data_train, wavelet='db4', level=2)
ema_train_denoised = ema(data_train_denoised, int(ema_window))
def rogers_satchell_volatility(data, periods):
    opens = data["open"].iloc[-periods:]
    highs = data["high"].iloc[-periods:]
    lows = data["low"].iloc[-periods:]
    closes = data["close"].iloc[-periods:]
    v1 = np.log(highs / closes)
    v2 = np.log(highs / opens)
    v3 = np.log(lows / closes)
    v4 = np.log(lows / opens)
    return np.sqrt(np.sum(v1 * v2 + v3 * v4) / periods)
periods = len(df_train)
ideal_w = rogers_satchell_volatility(df_train, periods=periods)
st.write(f"**Ideal w (volatility-influenced):** {ideal_w:.5f}")
labels_wavelet, label_dict_wavelet = auto_labeling(data_train_denoised, timestamps_train, ideal_w)
def generate_ground_truth_labels(data, horizon=1):
    gt = np.zeros(len(data))
    for i in range(len(data) - horizon):
        gt[i] = 1 if data[i + horizon] > data[i] else -1
    gt[-horizon:] = gt[len(data) - horizon - 1]
    return gt
gt_labels = generate_ground_truth_labels(data_train, horizon=1)
def evaluate_metrics(gt, pred):
    accuracy = accuracy_score(gt, pred)
    precision = precision_score(gt, pred, pos_label=1)
    recall = recall_score(gt, pred, pos_label=1)
    f1 = f1_score(gt, pred, pos_label=1)
    gt_bin = (gt == 1).astype(int)
    pred_bin = (pred == 1).astype(int)
    auc = roc_auc_score(gt_bin, pred_bin)
    return accuracy, precision, recall, f1, auc
metrics_wavelet = evaluate_metrics(gt_labels, labels_wavelet)
net_yield_wavelet = compute_investment_performance(data_train_denoised, labels_wavelet)
st.write("### Classification Metrics (Denoised Data with Volatility-Influenced w):")
st.write(f"Accuracy: {metrics_wavelet[0]:.4f}")
st.write(f"Precision: {metrics_wavelet[1]:.4f}")
st.write(f"Recall: {metrics_wavelet[2]:.4f}")
st.write(f"F1 Score: {metrics_wavelet[3]:.4f}")
st.write(f"AUC: {metrics_wavelet[4]:.4f}")
st.write(f"Net Yield: {net_yield_wavelet:.4f}")
fig_train, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
point_colors_orig = ['#10a4f4' if label == 1 else 'red' for label in labels_wavelet]
ax1.scatter(timestamps_train, data_train, c=point_colors_orig, s=5, label="Data Points", zorder=5)
ax1.plot(timestamps_train, data_train, color="gray", linewidth=0.8, alpha=0.7, label="Close Price")
ax1.plot(timestamps_train, ema_train, color="black", linewidth=0.5, label=f"EMA({ema_window})")
ax1.set_title(f"Original Data Auto Labeling (w = {ideal_w:.5f})")
ax1.set_ylabel("Price")
ax1.legend()
point_colors_denoised = ['#10a4f4' if label == 1 else 'red' for label in labels_wavelet]
ax2.scatter(timestamps_train, data_train_denoised, c=point_colors_denoised, s=5, label="Data Points", zorder=5)
ax2.plot(timestamps_train, data_train_denoised, color="gray", linewidth=0.8, alpha=0.7, label="Denoised Price")
ax2.plot(timestamps_train, ema_train_denoised, color="black", linewidth=0.5, label=f"EMA({ema_window})")
ax2.set_title(f"Denoised Data Auto Labeling (w = {ideal_w:.5f})")
ax2.set_xlabel("Timestamp")
ax2.set_ylabel("Price")
ax2.legend()
plt.xticks(rotation=30)
st.pyplot(fig_train)
