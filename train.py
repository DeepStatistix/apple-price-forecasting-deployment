# -*- coding: utf-8 -*-
# A-to-Z training runner (mask-aware)
# Adds season-aligned cross-validation (e.g., 2022→2023, 2023→2024), aggregates hyper-params,
# then refits on all data before TEST season and evaluates on the full TEST season.

import os, sys
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.ioff()

# -----------------------
# Environment flags
# -----------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["DISABLE_TF"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import sys, json, io, time, csv, math, random, platform, traceback, warnings, argparse
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# -----------------------
# Optional/third-party libs (silenced imports)
# -----------------------
class silence:
    def __enter__(self):
        self._stdout = io.StringIO(); self._stderr = io.StringIO()
        self._out = redirect_stdout(self._stdout); self._err = redirect_stderr(self._stderr)
        self._out.__enter__(); self._err.__enter__(); return self
    def __exit__(self, exc_type, exc, tb):
        self._err.__exit__(exc_type, exc, tb); self._out.__exit__(exc_type, exc, tb)

with silence():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        TF_OK = True
    except Exception:
        tf = None; TF_OK = False

with silence():
    try:
        from neuralprophet import NeuralProphet
        HAS_NP = True
    except Exception:
        HAS_NP = False

with silence():
    try:
        from tbats import TBATS
        HAS_TBATS = True
    except Exception:
        HAS_TBATS = False

with silence():
    try:
        from lightgbm import LGBMRegressor
        HAS_LGBM = True
    except Exception:
        HAS_LGBM = False

with silence():
    try:
        from statsmodels.tsa.stattools import acf
    except Exception:
        acf = None

with silence():
    try:
        from transformers import PatchTSTConfig, PatchTSTForPrediction
        HAS_HF_PATCHTST = True
    except Exception:
        HAS_HF_PATCHTST = False

# ---------- Darts (optional) ----------
HAS_DARTS = HAS_NBEATS = HAS_NHITS = False
HAS_PATCHTST_DARTS = False
try:
    import darts
    from darts import TimeSeries
    HAS_DARTS = True
    try:
        from darts.models import NBEATSModel
        HAS_NBEATS = True
    except Exception:
        HAS_NBEATS = False
    try:
        from darts.models import NHiTSModel
        HAS_NHITS = True
    except Exception:
        HAS_NHITS = False
    try:
        from darts.models import PatchTSTModel
        HAS_PATCHTST_DARTS = True
    except Exception:
        HAS_PATCHTST_DARTS = False
except Exception:
    pass

HAS_PATCHTST = HAS_PATCHTST_DARTS or HAS_HF_PATCHTST

# -----------------------
# Warnings & Seeds
# -----------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if TF_OK:
    try: tf.random.set_seed(RANDOM_SEED)
    except Exception: pass

# -----------------------
# GPU config (PyTorch)
# -----------------------
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if TORCH_DEVICE.type == "cuda":
    try: torch.set_float32_matmul_precision("medium")
    except Exception: pass

# -----------------------
# Global controls & CV config
# -----------------------
VAL_DAYS = 15                # used by 'precut_60d'
ONLINE_MODE = False             # teacher-forcing on test; keep False for fair OOS
SEQ_LENGTH_DEFAULT = 40         # for LSTM/Transformer
NN_BATCH = 64
EARLY_STOP_PATIENCE = 10

# Globals that will be updated per split/fold
scaler = None
scaler_stats = {}
TRAIN_OBS_Y = None

# Split globals (interpreted as: train < train_cut_dt; val = [train_cut_dt, val_end_dt); test = [TEST_START_DT, TEST_END_DT))
train_cut_dt = None
val_end_dt = None
TEST_START_DT = None
TEST_END_DT = None

# Forced hyper-params for final refit after CV (global!)
LGBM_FORCED_N_ESTIMATORS = None
RF_FORCED_MAX_DEPTH = None
RF_FORCED_MIN_LEAF = None

# -----------------------
# Experiment Logger
# -----------------------
class ExperimentLogger:
    def __init__(self, base_dir: str, run_name: str, config: dict):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(base_dir, f"{ts}_{run_name}")
        self.artifacts_dir = os.path.join(self.run_dir, "artifacts")
        self.models_dir = os.path.join(self.run_dir, "models")
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        self.pred_dir = os.path.join(self.run_dir, "predictions")
        self.figures_dir = os.path.join(self.run_dir, "figures")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        for d in [self.artifacts_dir, self.models_dir, self.metrics_dir, self.pred_dir, self.figures_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)
        self.config = config; self.summary = {}
        self._write_json("config.json", config)

        env = {"python": platform.python_version(), "platform": platform.platform(),
               "torch": torch.__version__, "tensorflow": (getattr(tf, "__version__", None) if TF_OK else None)}
        self._write_json("env.json", env)

        hw = {"torch_cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            hw.update({"cuda_version": torch.version.cuda, "gpu_count": torch.cuda.device_count(),
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
        self._write_json("hardware.json", hw)

        try:
            import pkg_resources
            pkgs = {dist.project_name: dist.version for dist in pkg_resources.working_set}
            self._write_json("packages.json", pkgs)
        except Exception:
            pass

        self.time_summary_path = os.path.join(self.metrics_dir, "time_summary.csv")
        self.metrics_summary_path = os.path.join(self.metrics_dir, "metrics_summary.csv")
        self.params_summary_path = os.path.join(self.metrics_dir, "params_summary.csv")
        self._init_csv(self.time_summary_path, ["model", "total_seconds", "device", "lib"])
        self._init_csv(self.metrics_summary_path, ["model", "run_seconds", "MSE", "RMSE", "MAE", "MdAE", "RMSLE", "MAPE_%",
                                                   "sMAPE_%", "R2", "Corr", "MBE", "PBIAS_%", "NRMSE_mean",
                                                   "NRMSE_range", "Theils_U1", "Theils_U2", "Directional_Accuracy_%", "MASE"])
        self._init_csv(self.params_summary_path, ["model", "param", "value"])

    def _write_json(self, name, obj):
        path = os.path.join(self.run_dir, name)
        with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2, default=str)
        return path

    def _init_csv(self, path, headers):
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(headers)

    def _append_csv(self, path, row_dict):
        key = row_dict.get("model")
        df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        if key is not None and "model" in df.columns and not df.empty:
            df = df[df["model"] != key]
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
        df.to_csv(path, index=False)

    def log_params(self, model_name: str, params: dict):
        path = os.path.join(self.metrics_dir, f"{model_name}_params.json")
        with open(path, "w", encoding="utf-8") as f: json.dump(params, f, indent=2, default=str)
        for k, v in params.items():
            self._append_csv(self.params_summary_path, {"model": model_name, "param": str(k), "value": json.dumps(v, default=str)})

    def log_metrics(self, model_name: str, metrics: dict):
        path = os.path.join(self.metrics_dir, f"{model_name}_metrics.json")
        with open(path, "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2, default=str)
        self.summary.setdefault(model_name, {}).update(metrics)
        row = {"model": model_name}
        row.update({k: metrics.get(k, np.nan) for k in
                    ["run_seconds","MSE","RMSE","MAE","MdAE","RMSLE","MAPE_%","sMAPE_%","R2","Corr","MBE",
                     "PBIAS_%","NRMSE_mean","NRMSE_range","Theils_U1","Theils_U2","Directional_Accuracy_%","MASE"]})
        self._append_csv(self.metrics_summary_path, row)

    def log_time_row(self, model_name: str, total_seconds: float, device: str, lib: str):
        self._append_csv(self.time_summary_path, {"model": model_name, "total_seconds": float(total_seconds), "device": device, "lib": lib})

    def save_predictions_table(self, name: str, df: pd.DataFrame):
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.pred_dir, f"{name}_{ts_tag}.csv")
        df.to_csv(path, index=False); return path

    def save_model(self, model_name: str, model_obj, kind: str):
        ts_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = os.path.join(self.models_dir, f"{model_name}_{ts_tag}")
        try:
            if kind == "sklearn":
                path = base + ".joblib"; joblib.dump(model_obj, path)
            elif kind == "tf":
                path = base + ".h5"; model_obj.save(path)
            elif kind == "torch":
                path = base + ".pt"; torch.save(model_obj.state_dict(), path)
            elif kind == "darts":
                path = base + ".darts.pt"; model_obj.save(path)
            else:
                path = base + ".bin"; open(path, "wb").close()
        except Exception as e:
            path = base + ".ERR"
            with open(path, "w", encoding="utf-8") as f: f.write(str(e))
        return path

    def log_exception(self, model_name: str, e: Exception):
        tb = traceback.format_exc()
        path = os.path.join(self.logs_dir, f"{model_name}_error.txt")
        with open(path, "w", encoding="utf-8") as f: f.write(tb)
        print(f"[ERROR] {model_name}: {e}")

    def finalize(self):
        self._write_json("summary.json", self.summary)
        print(f"\nAll artifacts saved under: {self.run_dir}")

# -----------------------
# Load & prepare data  (mask-aware)
# -----------------------
data_path = r'D:\Git Projects\marketdata\data\raw\processed\Pulwama\Pachhar\American_B_dataset.csv'
raw = pd.read_csv(data_path)

raw = raw.rename(columns={"Date": "ds", "Avg Price (per kg)": "y", "Mask": "mask"})
raw['y'] = pd.to_numeric(raw['y'], errors='coerce')
raw['ds'] = pd.to_datetime(raw['ds'])
raw['mask'] = raw['mask'].fillna(0).astype(int)

full_data = raw.sort_values('ds').reset_index(drop=True).copy()
available_data = full_data.loc[(full_data['mask'] == 1) & (full_data['y'].notna())].sort_values('ds').reset_index(drop=True).copy()

# ---------- derive dataset tag (market / commodity / optional grade) ----------
import re

def infer_dataset_tag(path: str):
    # Market = parent folder name (e.g., Azadpur)
    market = os.path.basename(os.path.dirname(path)).strip() or "UnknownMarket"

    # File base without extension or trailing "_dataset"
    fname = os.path.splitext(os.path.basename(path))[0]
    base  = re.sub(r'(_dataset)?$', '', fname, flags=re.IGNORECASE)

    # Expect formats like "American_A", "American_B", or just "American"
    m = re.match(r'^(?P<commodity>.+?)(?:_(?P<grade>[A-D]))?$', base, flags=re.IGNORECASE)
    commodity = (m.group('commodity') if m else base).strip().replace('-', '_').replace(' ', '_')
    grade = (m.group('grade').upper() if m and m.group('grade') else None)

    # Optional fallback: if a Grade column exists in the CSV, prefer it
    # (keeps A/B/C… if consistently present; ignore if mixed/empty)
    if grade is None and 'Grade' in raw.columns:
        try:
            g = str(raw['Grade'].dropna().mode().iloc[0]).strip().upper()
            if re.fullmatch(r'[A-D]', g): grade = g
        except Exception:
            pass

    dataset_tag = "_".join([x for x in (market, commodity, grade) if x])
    return market, commodity, grade, dataset_tag

MARKET, COMMODITY, GRADE, DATASET_TAG = infer_dataset_tag(data_path)
if GRADE:
    print(f"[INFO] Detected dataset tag: {DATASET_TAG} (grade {GRADE})")
else:
    print(f"[INFO] Detected dataset tag: {DATASET_TAG} (no grade token found)")

# -----------------------
# Plotting
# -----------------------
def _save_plot(dates, y_true, y_pred, out_path, title):
    import matplotlib.dates as mdates
    fig = plt.figure(figsize=(10, 6)); ax = plt.gca()
    ax.plot(dates, y_true, label='Actual', linewidth=2.0)
    ax.plot(dates, y_pred, label='Predicted', linestyle='--', linewidth=1.6)
    ax.set_title(title); ax.set_xlabel('Time'); ax.set_ylabel('Price (Rs/kg)'); ax.legend()

    # Force monthly ticks with month abbreviations
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Jan, Feb, ...

    # (optional) show the year on a second, small label row
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('\n%Y'))
    ax.tick_params(axis='x', which='minor', pad=10, labelsize=9)

    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# -----------------------
# Helpers: splitting and sequences
# -----------------------
def _season_bounds(year: int, season_start: str, season_end: str):
    """
    Return (start_dt, end_dt_exclusive) for a season that may cross year boundary.
    season_start/end are 'MM-DD'.
    """
    start_dt = pd.Timestamp(f"{year}-{season_start}")
    # end lives in next year if end month-day is before start month-day
    end_year = year if pd.Timestamp(f"2000-{season_end}") > pd.Timestamp(f"2000-{season_start}") else year + 1
    end_dt = pd.Timestamp(f"{end_year}-{season_end}") + pd.Timedelta(days=1)  # exclusive
    return start_dt, end_dt

def _rebuild_splits(cut_start_dt: pd.Timestamp,
                    cut_end_dt: pd.Timestamp,
                    test_start_dt: pd.Timestamp = None,
                    test_end_dt: pd.Timestamp = None):
    """
    Rebuild globals for a split where:
      - Train: ds < cut_start_dt
      - Val  : cut_start_dt <= ds < cut_end_dt
      - Test : ds >= test_start_dt (and < test_end_dt if provided)
    Also (re)fit the global scaler on observed *training* targets only.
    """
    global train_cut_dt, val_end_dt, TEST_START_DT, TEST_END_DT
    global scaler, scaler_stats, TRAIN_OBS_Y
    global train_full_pre, val_full, test_full, trainval_full

    train_cut_dt = pd.to_datetime(cut_start_dt)
    val_end_dt = pd.to_datetime(cut_end_dt)
    TEST_START_DT = pd.to_datetime(test_start_dt) if test_start_dt is not None else val_end_dt
    TEST_END_DT = pd.to_datetime(test_end_dt) if test_end_dt is not None else None

    # Calendar frames
    train_full_pre = full_data.loc[full_data['ds'] < train_cut_dt].copy()
    val_full = full_data.loc[(full_data['ds'] >= train_cut_dt) & (full_data['ds'] < val_end_dt)].copy()
    if TEST_END_DT is None:
        test_full = full_data.loc[full_data['ds'] >= TEST_START_DT].copy()
    else:
        test_full = full_data.loc[(full_data['ds'] >= TEST_START_DT) & (full_data['ds'] < TEST_END_DT)].copy()

    # Observed frames
    train_obs = available_data.loc[available_data['ds'] < train_cut_dt].copy()
    val_obs   = available_data.loc[(available_data['ds'] >= train_cut_dt) & (available_data['ds'] < val_end_dt)].copy()
    if TEST_END_DT is None:
        test_obs  = available_data.loc[available_data['ds'] >= TEST_START_DT].copy()
    else:
        test_obs  = available_data.loc[(available_data['ds'] >= TEST_START_DT) & (available_data['ds'] < TEST_END_DT)].copy()

    # Fit scaler on observed training only
    global_scaler = StandardScaler()
    if not train_obs.empty:
        train_obs['y_scaled'] = global_scaler.fit_transform(train_obs[['y']])
    else:
        global_scaler.fit(np.array([[0.0]]))
        train_obs['y_scaled'] = np.nan

    # make global
    global scaler
    scaler = global_scaler

    val_obs['y_scaled']  = scaler.transform(val_obs[['y']]) if not val_obs.empty else np.nan
    test_obs['y_scaled'] = scaler.transform(test_obs[['y']]) if not test_obs.empty else np.nan

    # Merge y_scaled back to calendar frames
    train_full_pre = train_full_pre.merge(train_obs[['ds','y_scaled']], on='ds', how='left')
    val_full = val_full.merge(val_obs[['ds','y_scaled']], on='ds', how='left')
    test_full = test_full.merge(test_obs[['ds','y_scaled']], on='ds', how='left')

    trainval_full = pd.concat([train_full_pre, val_full], ignore_index=True)

    TRAIN_OBS_Y = train_obs['y'].values  # for MASE
    scaler_stats = {"mean": float(scaler.mean_[0]), "scale": float(scaler.scale_[0])}

def split_after_cut_on_calendar(df):
    """Return (train_pre, val_after, test_after_val, val_end_dt)."""
    trm = df[df['ds'] < train_cut_dt].copy()
    va  = df[(df['ds'] >= train_cut_dt) & (df['ds'] < val_end_dt)].copy()
    te  = df[df['ds'] >= (TEST_START_DT if TEST_START_DT is not None else val_end_dt)].copy()
    if TEST_END_DT is not None:
        te = te[te['ds'] < TEST_END_DT].copy()
    return trm, va, te, val_end_dt

def create_sequences_for_window(df, seq_length, start_date, end_date, feature_cols=('y_scaled_filled','mask')):
    d = df.copy()
    d['y_scaled_filled'] = d['y_scaled'].ffill().fillna(0.0)
    if 'mask' not in d.columns: d['mask'] = 1
    vals = d[list(feature_cols)].to_numpy(np.float32)
    yraw = d['y_scaled'].to_numpy(np.float32)
    msk  = d['mask'].to_numpy(np.int32)
    dates = d['ds'].to_numpy()
    X, y = [], []
    for i in range(len(d) - seq_length):
        j = i + seq_length
        if not (start_date <= dates[j] < end_date):
            continue
        if msk[j] != 1 or not np.isfinite(yraw[j]):  # label must be an observed point
            continue
        X.append(vals[i:j, :]); y.append(yraw[j])
    if not X: return np.array([]), np.array([])
    return np.stack(X, 0), np.asarray(y)

# -------- Safe inverse-scale everywhere --------
def inv_scale_safe(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(-1, 1)
    if a.shape[0] == 0:
        return np.array([], dtype=float)
    return scaler.inverse_transform(a).ravel()

def roll_predict_nn_full(model, trainval_df, test_df, seq_length, framework='tf'):
    """Start rolling exactly at test start (right after val), seed with last seq_length steps."""
    # No-op when no test horizon (CV folds)
    if len(test_df) == 0:
        return np.array([], dtype=float)

    comb = pd.concat([trainval_df[['y_scaled','mask']], test_df[['y_scaled','mask']]], ignore_index=True)
    comb['y_scaled_filled'] = comb['y_scaled'].ffill().fillna(0.0)
    start_idx = len(trainval_df)  # first test index (after val / at test start)
    w_start = max(0, start_idx - seq_length)
    window = comb.iloc[w_start:start_idx][['y_scaled_filled','mask']].to_numpy(dtype=np.float32)
    if len(window) < seq_length:
        pad = np.repeat(window[:1, :], repeats=(seq_length - len(window)), axis=0) if len(window)>0 else np.zeros((seq_length, 2), np.float32)
        window = np.vstack([pad, window]).astype(np.float32)

    preds = []
    for t in range(len(test_df)):
        x = window[None, :, :]
        if framework == 'tf':
            yhat_s = float(model.predict(x, verbose=0).reshape(-1)[0])
        else:
            with torch.no_grad():
                tens = torch.tensor(x, dtype=torch.float32, device=TORCH_DEVICE)
                yhat_s = float(model(tens).detach().cpu().numpy().reshape(-1)[0])
        preds.append(yhat_s)

        idx = start_idx + t
        if ONLINE_MODE and np.isfinite(comb['y_scaled'].iloc[idx]) and int(test_df['mask'].iloc[t]) == 1:
            next_val = float(test_df['y_scaled'].iloc[t])
        else:
            next_val = yhat_s
        new_row = np.array([[next_val, float(test_df['mask'].iloc[t])]], dtype=np.float32)
        window = np.vstack([window[1:], new_row])

    preds = np.asarray(preds, dtype=np.float32)
    return inv_scale_safe(preds)

def _safe_chunk_lengths(L, h, desired_out=14, min_in=8, max_in=192):
    if L < 5:
        return None, None  # far too short
    out_len = max(1, min(desired_out, h, max(1, L // 8)))
    in_len = max(min_in, min(max_in, L - out_len - 1))
    while (in_len + out_len + 1) > L and out_len > 1:
        out_len -= 1
        in_len = max(min_in, min(max_in, L - out_len - 1))
    if (in_len + out_len + 1) > L or in_len < min_in or out_len < 1:
        return None, None
    return int(in_len), int(out_len)

def build_lag_frame(df_all, max_lag, use_mask=True):
    d = df_all.copy()
    d['y_scaled_filled'] = d['y_scaled'].ffill()
    for k in range(1, max_lag+1):
        d[f'y_lag{k}'] = d['y_scaled_filled'].shift(k)
        if use_mask and 'mask' in d.columns:
            d[f'mask_lag{k}'] = d['mask'].shift(k)
    feats = [f'y_lag{i}' for i in range(1, max_lag+1)]
    if use_mask and 'mask' in d.columns:
        feats += [f'mask_lag{i}' for i in range(1, max_lag+1)]
    return d, feats

# ---------- Darts helpers ----------
def _choose_window_lengths(L, out_len_default=14, min_in=16, max_in=192):
    out_len = min(out_len_default, max(1, L // 8))
    in_len  = min(max_in, max(min_in, L - out_len - 1))
    if in_len + out_len >= L:
        in_len = max(min_in, L - out_len - 1)
        if in_len < min_in:
            in_len = max(4, L // 2); out_len = max(1, L - in_len - 1)
    return int(in_len), int(out_len)

def _ts_from_scaled(df):
    d = df[['ds', 'y_scaled']].copy()
    zero_scaled_value = (0 - scaler.mean_[0]) / scaler.scale_[0]
    d['y_scaled'] = d['y_scaled'].ffill().fillna(zero_scaled_value)
    return TimeSeries.from_dataframe(d, time_col='ds', value_cols='y_scaled', freq='D')

# -----------------------
# TensorFlow config
# -----------------------
try:
    TF_HAS_GPU = TF_OK and len(tf.config.list_physical_devices('GPU')) > 0
except Exception:
    TF_HAS_GPU = False
TF_DEVICE = '/GPU:0' if TF_HAS_GPU else '/CPU:0'
if TF_HAS_GPU:
    with silence():
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            pass

# -----------------------
# Conformal intervals
# -----------------------
def conformal_intervals(val_true, val_pred, test_pred, alpha=0.1):
    vt = np.asarray(val_true, float).ravel()
    vp = np.asarray(val_pred, float).ravel()
    tp = np.asarray(test_pred, float).ravel()
    n = min(len(vt), len(vp))
    if n == 0: return np.full_like(tp, np.nan), np.full_like(tp, np.nan), np.nan
    resid = np.abs(vt[-n:] - vp[-n:])
    q = float(np.quantile(resid, 1.0 - alpha))
    return tp - q, tp + q, q

# -----------------------
# Models
# -----------------------
MASK_AWARE_MODELS = {'lgbm', 'rf', 'lstm', 'transformer'}
UNIVARIATE_ONLY   = {'nbeats', 'nhits', 'patchtst', 'tbats', 'neuralprophet'}

def dataset_for(model_name: str):
    # Train on train+val, predict on test (test starts at TEST_START_DT or val_end_dt)
    return trainval_full.copy(), test_full.copy()

# --- NeuralProphet ---
def _np_make(**kwargs):
    safe = dict(epochs=100, batch_size=32, learning_rate=1e-3, daily_seasonality=False, n_lags=0, n_forecasts=1)
    safe.update(kwargs)
    try: return NeuralProphet(**safe)
    except Exception: return NeuralProphet()

def _np_yhat(df):
    if 'yhat1' in df.columns: return df['yhat1'].values
    if 'yhat'  in df.columns: return df['yhat'].values
    cols = [c for c in df.columns if c.startswith('yhat')]
    return df[cols[0]].values if cols else np.array([])

def neuralprophet_model(trainval_df, test_df, logger=None):
    if not HAS_NP:
        print("[WARN] NeuralProphet not available → NaNs")
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"neuralprophet"}

    # Split for train/val inside the combined train+val frame
    trm = trainval_df[trainval_df['ds'] < train_cut_dt].copy()
    va  = trainval_df[(trainval_df['ds'] >= train_cut_dt) & (trainval_df['ds'] < val_end_dt)].copy()

    # Clean training history for NP (no NaNs)
    train_hist = trm[['ds','y']].dropna().sort_values('ds')
    if train_hist.empty:
        print("[WARN] NeuralProphet: empty TRAIN after dropna → skipping")
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"neuralprophet"}

    # ---- fit on TRAIN and get VAL predictions (if any) ----
    with silence():
        try:
            m_tr = _np_make(early_stopping=True)
        except TypeError:
            m_tr = _np_make()
        m_tr.fit(train_hist, freq='D')

        if len(va) > 0:
            future_val = m_tr.make_future_dataframe(
                train_hist, periods=len(va), n_historic_predictions=False
            )
            fcst_val = m_tr.predict(future_val)
            val_pred = _np_yhat(fcst_val)[-len(va):]
        else:
            val_pred = np.array([])

    # ---- (optional) refit on TRAIN+VAL and get TEST predictions ----
    test_pred = np.array([])
    if len(test_df) > 0:
        trainval_hist = (
            pd.concat([trm, va], ignore_index=True)[['ds','y']]
            .dropna().sort_values('ds')
        )
        if not trainval_hist.empty:
            with silence():
                m_full = _np_make()
                m_full.fit(trainval_hist, freq='D')
                future_test = m_full.make_future_dataframe(
                    trainval_hist, periods=len(test_df), n_historic_predictions=False
                )
                fcst_test = m_full.predict(future_test)
                test_pred = _np_yhat(fcst_test)[-len(test_df):]
            if logger:
                logger.log_params("neuralprophet", {"epochs":100,"batch_size":32,"lr":1e-3,"lib":"neuralprophet"})
                try: logger.save_model("neuralprophet", m_full, kind="rawdir")
                except Exception: pass

    vtrue = va['y'].to_numpy() if len(va) > 0 else np.array([])
    return test_pred, (vtrue, val_pred), {"lib":"neuralprophet"}

# --- TBATS ---
# --- TBATS (faster) ---
def tbats_model(trainval_df, test_df, logger=None, fast=True, max_train_len=1500):
    """
    Faster TBATS:
    - keeps the best model from the candidate search (no extra re-fit for validation)
    - optional ARMA errors (fast=True -> use_arma_errors=False)
    - trims very long histories to the last `max_train_len` points
    - uses a leaner seasonal candidate list for daily data
    """
    if not HAS_TBATS:
        print("[WARN] TBATS not available → NaNs")
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"tbats"}

    trm = trainval_df[trainval_df['ds'] < train_cut_dt].copy()
    va  = trainval_df[(trainval_df['ds'] >= train_cut_dt) & (trainval_df['ds'] < val_end_dt)].copy()

    # y arrays (drop NaNs)
    y_trm_full = trm['y'].dropna().to_numpy(float)
    y_va       = va['y'].dropna().to_numpy(float)

    # Cap history length to speed up
    if max_train_len is not None and len(y_trm_full) > max_train_len:
        y_trm = y_trm_full[-max_train_len:]
    else:
        y_trm = y_trm_full

    # Decide period candidates quickly from sampling step
    d = trm[['ds','y']].dropna().sort_values('ds')
    if len(d) >= 3:
        step_days = int(np.median(np.diff(d['ds'].values).astype('timedelta64[D]').astype(int)))
    else:
        step_days = 1

    # Leaner candidates by default
    if step_days >= 27:
        candidates = [[12]]  # roughly monthly series
    elif 5 <= step_days <= 9:
        candidates = [[7]]   # weekly
    else:
        candidates = [[7]]   # daily → start simple; add [7,30] only if needed

    use_arma = (not fast)  # fast=True disables ARMA errors (big speed win)

    best_sp = None
    best_mae = np.inf
    best_model = None
    best_val_pred = None

    # If there is no validation window, skip the search
    do_search = len(va) > 0

    # Candidate search on TRAIN only
    if do_search:
        tried_any = False
        for sp in candidates:
            try:
                with silence():
                    m = TBATS(seasonal_periods=sp, use_arma_errors=use_arma, use_box_cox=False, n_jobs=1).fit(y_trm)
                    pred = np.asarray(m.forecast(steps=len(va)), float) if len(va) > 0 else np.array([], float)
                mae = np.mean(np.abs(pred - y_va)) if len(pred) == len(y_va) and len(pred) > 0 else np.inf
                tried_any = True
                if mae < best_mae:
                    best_mae = mae; best_sp = sp; best_model = m; best_val_pred = pred
            except Exception:
                continue

        # If nothing worked, add a second-chance candidate [7,30] for daily data
        if (not tried_any or best_sp is None) and step_days < 5:
            sp = [7, 30]
            try:
                with silence():
                    m = TBATS(seasonal_periods=sp, use_arma_errors=use_arma, use_box_cox=False, n_jobs=1).fit(y_trm)
                    pred = np.asarray(m.forecast(steps=len(va)), float) if len(va) > 0 else np.array([], float)
                mae = np.mean(np.abs(pred - y_va)) if len(pred) == len(y_va) and len(pred) > 0 else np.inf
                best_mae = mae; best_sp = sp; best_model = m; best_val_pred = pred
            except Exception:
                pass
    else:
        # No validation: just pick a sensible default
        best_sp = candidates[0]

    if best_sp is None:
        best_sp = candidates[-1]

    # If search didn’t leave us with a fitted train model, fit one now (train only)
    if best_model is None:
        with silence():
            best_model = TBATS(seasonal_periods=best_sp, use_arma_errors=use_arma, use_box_cox=False, n_jobs=1).fit(y_trm)
        best_val_pred = (np.asarray(best_model.forecast(steps=len(va)), float) if len(va) > 0 else np.array([], float))

    # Fit on TRAIN+VAL for final TEST forecast (cannot avoid this one)
    y_full_pretest = pd.concat([trm, va])['y'].dropna().to_numpy(float)
    if max_train_len is not None and len(y_full_pretest) > max_train_len:
        y_full_pretest = y_full_pretest[-max_train_len:]

    try:
        with silence():
            m_full = TBATS(seasonal_periods=best_sp, use_arma_errors=use_arma, use_box_cox=False, n_jobs=1).fit(y_full_pretest)
            test_pred = (np.asarray(m_full.forecast(steps=len(test_df)), float) if len(test_df) > 0 else np.array([], float))
    except Exception as e:
        if logger: logger.log_exception("tbats", e)
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"tbats"}

    # Validation outputs (from the kept best train-only model)
    val_true = va['y'].values if len(va) > 0 else np.array([])
    val_pred = best_val_pred if best_val_pred is not None else (np.array([], float))

    if logger:
        logger.log_params("tbats", {
            "seasonal_periods": best_sp,
            "use_arma_errors": bool(use_arma),
            "use_box_cox": False,
            "n_jobs": 1,
            "fast_mode": bool(fast),
            "max_train_len": int(max_train_len) if max_train_len is not None else None,
            "lib": "tbats"
        })
        try: logger.save_model("tbats", m_full, kind="rawdir")
        except Exception: pass

    return test_pred, (val_true, val_pred), {"lib":"tbats"}


# --- LightGBM (mask-aware) ---
def lgbm_model(trainval_df, test_df, logger=None):
    """
    Two modes:
      1) With validation window (len(idx_va)>0): early stopping on VAL, then refit on TRAIN+VAL with best_n.
      2) No validation (final refit after CV): use forced n_estimators if provided, train on all ds < TEST_START_DT.
    """
    if not HAS_LGBM:
        print("[WARN] LightGBM not available → NaNs")
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib": "sklearn"}

    try:
        from lightgbm.callback import early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
    except Exception:
        lgb_early_stopping = None
        lgb_log_evaluation = None

    max_lag = 40
    td_all, feats = build_lag_frame(trainval_df, max_lag, use_mask=True)
    label_cond = td_all['y_scaled'].notna() & (td_all['mask'] == 1)

    idx_tr   = td_all.index[(td_all['ds'] <  train_cut_dt) & label_cond]
    idx_va   = td_all.index[(td_all['ds'] >= train_cut_dt) & (td_all['ds'] < val_end_dt) & label_cond]
    idx_trva = td_all.index[(td_all['ds'] <  val_end_dt) & label_cond]
    # For final refit we want everything < TEST_START_DT
    idx_pretest_all = td_all.index[(td_all['ds'] < (TEST_START_DT if TEST_START_DT is not None else val_end_dt)) & label_cond]

    params = dict(
        n_estimators=100,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED
    )
    if TORCH_DEVICE.type == "cuda":
        params.update(device_type='gpu')

    best_n = None
    if len(idx_va) > 0 and len(idx_tr) > 0:
        # Early stopping on validation
        callbacks = []
        if lgb_early_stopping is not None:
            callbacks.append(lgb_early_stopping(stopping_rounds=100))
        if lgb_log_evaluation is not None:
            callbacks.append(lgb_log_evaluation(period=0))  # silence

        model = LGBMRegressor(**params)
        model.fit(td_all.loc[idx_tr, feats], td_all.loc[idx_tr, 'y_scaled'],
                  eval_set=[(td_all.loc[idx_va, feats], td_all.loc[idx_va, 'y_scaled'])],
                  eval_metric='l1', callbacks=callbacks)
        best_n = int(getattr(model, "best_iteration_", params['n_estimators']) or params['n_estimators'])

        # Refit on TRAIN+VAL
        model_full = LGBMRegressor(**{**params, "n_estimators": best_n})
        model_full.fit(td_all.loc[idx_trva, feats], td_all.loc[idx_trva, 'y_scaled'])

        # Validation pairs
        val_pred_s = model.predict(td_all.loc[idx_va, feats]) if len(idx_va)>0 else np.array([])
        val_true   = inv_scale_safe(td_all.loc[idx_va, 'y_scaled'].to_numpy()) if len(idx_va)>0 else np.array([])
        val_pred   = inv_scale_safe(val_pred_s) if val_pred_s is not None else np.array([])
    else:
        # Final refit: no validation. Use forced n if available.
        forced_n = LGBM_FORCED_N_ESTIMATORS if LGBM_FORCED_N_ESTIMATORS is not None else params['n_estimators']
        best_n = int(forced_n)
        model_full = LGBMRegressor(**{**params, "n_estimators": best_n})
        if len(idx_pretest_all) == 0:
            print("[WARN] LGBM: no training rows before test start")
            return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"sklearn","best_n":best_n}
        model_full.fit(td_all.loc[idx_pretest_all, feats], td_all.loc[idx_pretest_all, 'y_scaled'])
        val_true, val_pred = np.array([]), np.array([])

    # Recursive test forecast from exact test start
    full_for_seed = pd.concat([trainval_df[['y_scaled','mask']], test_df[['y_scaled','mask']]], ignore_index=True)
    y_hist = full_for_seed['y_scaled'].ffill().to_numpy()
    y_hist_trva = y_hist[:len(trainval_df)]
    w = y_hist_trva[-max_lag:]
    if w.size < max_lag:
        pad_val = w[0] if w.size else 0.0
        w = np.concatenate([np.repeat(pad_val, max_lag - w.size), w]).astype(np.float32)
    else:
        w = w.astype(np.float32)

    mask_hist = full_for_seed['mask'].to_numpy().astype(int)
    preds_s = np.zeros(len(test_df), np.float32)

    for t in range(len(test_df)):
        base_idx = len(trainval_df) - max_lag + t
        xlags = np.array([w[-k] for k in range(1, max_lag + 1)], np.float32)
        mask_window = np.array([mask_hist[base_idx + k] for k in range(0, max_lag)], np.float32)
        x = np.concatenate([xlags, mask_window], axis=0)

        yhat_s = float(model_full.predict(x.reshape(1, -1))[0]); preds_s[t] = yhat_s
        next_val = (float(test_df['y_scaled'].iloc[t]) if (ONLINE_MODE and int(test_df['mask'].iloc[t]) == 1
                     and np.isfinite(test_df['y_scaled'].iloc[t])) else yhat_s)
        w = np.concatenate([w[1:], [next_val]]).astype(np.float32)

    y_pred = inv_scale_safe(preds_s)

    if logger:
        logger.log_params("lgbm", {"max_lag": 40, "best_n": int(best_n) if best_n is not None else None, "lib": "sklearn"})
        try: logger.save_model("lgbm", model_full, kind="sklearn")
        except Exception: pass

    return y_pred, (val_true, val_pred), {"lib": "sklearn", "best_n": int(best_n) if best_n is not None else None}

# --- RandomForest ---
def random_forest_model(trainval_df, test_df, logger=None):
    max_lag = 40
    td_all, feats = build_lag_frame(trainval_df, max_lag, use_mask=True)

    label_cond = td_all['y_scaled'].notna() & (td_all['mask'] == 1)
    idx_tr = td_all.index[(td_all['ds'] < train_cut_dt) & label_cond]
    idx_va = td_all.index[(td_all['ds'] >= train_cut_dt) & (td_all['ds'] < val_end_dt) & label_cond]
    idx_trva = td_all.index[(td_all['ds'] < val_end_dt) & label_cond]
    idx_pretest_all = td_all.index[(td_all['ds'] < (TEST_START_DT if TEST_START_DT is not None else val_end_dt)) & label_cond]

    best_cfg = (None, 1)  # (max_depth, min_samples_leaf)
    if len(idx_tr) > 0 and len(idx_va) > 0:
        grid_depth = [None, 6, 12, 24]
        grid_leaf  = [1, 3]
        best_mae = np.inf
        for dpt in grid_depth:
            for leaf in grid_leaf:
                rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1,
                                           max_depth=dpt, min_samples_leaf=leaf)
                rf.fit(td_all.loc[idx_tr, feats], td_all.loc[idx_tr, 'y_scaled'])
                vp = rf.predict(td_all.loc[idx_va, feats])
                mae = np.mean(np.abs(vp - td_all.loc[idx_va, 'y_scaled'].to_numpy()))
                if mae < best_mae:
                    best_mae = mae; best_cfg = (dpt, leaf)
        dpt, leaf = best_cfg
        rf_full = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1,
                                        max_depth=dpt, min_samples_leaf=leaf)
        rf_full.fit(td_all.loc[idx_trva, feats], td_all.loc[idx_trva, 'y_scaled'])

        val_pred_s = rf_full.predict(td_all.loc[idx_va, feats]) if len(idx_va)>0 else np.array([])
        val_true   = inv_scale_safe(td_all.loc[idx_va, 'y_scaled'].to_numpy()) if len(idx_va)>0 else np.array([])
        val_pred   = inv_scale_safe(val_pred_s) if len(idx_va)>0 else np.array([])
    else:
        # final refit without validation, use forced if provided
        dpt = RF_FORCED_MAX_DEPTH
        leaf = RF_FORCED_MIN_LEAF
        if dpt is None: dpt = None
        if leaf is None: leaf = 1
        rf_full = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1,
                                        max_depth=dpt, min_samples_leaf=leaf)
        if len(idx_pretest_all) == 0:
            print("[WARN] RF: no training rows before test start")
            return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"sklearn","best_max_depth":dpt,"best_min_leaf":leaf}
        rf_full.fit(td_all.loc[idx_pretest_all, feats], td_all.loc[idx_pretest_all, 'y_scaled'])
        val_true, val_pred = np.array([]), np.array([])

    # recursive forecast
    full_for_seed = pd.concat([trainval_df[['y_scaled','mask']], test_df[['y_scaled','mask']]], ignore_index=True)
    w = full_for_seed['y_scaled'].ffill().to_numpy()[:len(trainval_df)][-max_lag:]
    if w.size < max_lag:
        w = np.concatenate([np.repeat(w[:1], max_lag - w.size), w]).astype(np.float32)
    mask_hist = full_for_seed['mask'].to_numpy().astype(int)
    preds_s = np.zeros(len(test_df), np.float32)

    for t in range(len(test_df)):
        base_idx = len(trainval_df) - max_lag + t
        xlags = np.array([w[-k] for k in range(1, max_lag+1)], np.float32)
        mask_window = np.array([mask_hist[base_idx + k] for k in range(0, max_lag)], np.float32)
        x = np.concatenate([xlags, mask_window], axis=0)
        yhat_s = float(rf_full.predict(x.reshape(1,-1))[0]); preds_s[t] = yhat_s
        next_val = float(test_df['y_scaled'].iloc[t]) if (ONLINE_MODE and int(test_df['mask'].iloc[t])==1 and np.isfinite(test_df['y_scaled'].iloc[t])) else yhat_s
        w = np.concatenate([w[1:], [next_val]]).astype(np.float32)

    y_pred = inv_scale_safe(preds_s)

    if logger:
        logger.log_params("rf", {"max_lag":max_lag, "best_max_depth":best_cfg[0], "best_min_leaf":best_cfg[1], "lib":"sklearn"})
        try: logger.save_model("rf", rf_full, kind="sklearn")
        except Exception: pass

    return y_pred, (val_true, val_pred), {"lib":"sklearn","best_max_depth":best_cfg[0],"best_min_leaf":best_cfg[1]}

# --- LSTM (TF) ---
def lstm_model(trainval_df, test_df, logger=None):
    if not TF_OK:
        print("[WARN] TensorFlow not available → NaNs")
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"tf"}

    seq_length = SEQ_LENGTH_DEFAULT
    Xtr, ytr = create_sequences_for_window(trainval_df, seq_length, start_date=trainval_df['ds'].min(), end_date=train_cut_dt)
    Xv,  yv  = create_sequences_for_window(trainval_df, seq_length, start_date=train_cut_dt, end_date=val_end_dt)

    with tf.device(TF_DEVICE):
        model = Sequential([
            LSTM(128, activation='tanh', input_shape=(seq_length, Xtr.shape[-1] if len(Xtr)>0 else 2)),
            Dense(1, dtype='float32')
        ])
        model.compile(optimizer='adam', loss='mse')
        t0 = time.time()
        if len(Xtr) > 0 and len(Xv) > 0:
            early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
            model.fit(Xtr, ytr, epochs=100, batch_size=NN_BATCH, verbose=0, validation_data=(Xv, yv), callbacks=[early])
        else:
            # final refit: train on all data up to test start with fixed epochs
            if len(Xtr) == 0:
                return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"tf", "seq_length":seq_length}
            model.fit(Xtr, ytr, epochs=60, batch_size=NN_BATCH, verbose=0)
        train_seconds = time.time() - t0

    if len(Xv) > 0:
        vpred_s = model.predict(Xv, verbose=0).reshape(-1)
        vpred = inv_scale_safe(vpred_s)
        vtrue = inv_scale_safe(yv)
    else:
        vpred = np.array([]); vtrue = np.array([])

    test_pred = roll_predict_nn_full(model, trainval_df, test_df, seq_length, framework='tf')

    if logger:
        logger.log_params("lstm", {"seq_length":seq_length, "units":128, "epochs":"ES_or_60", "batch_size":NN_BATCH,
                                   "train_seconds":float(train_seconds), "lib":"tf"})
        try: logger.save_model("lstm", model, kind="tf")
        except Exception: pass
    return test_pred, (vtrue, vpred), {"lib":"tf", "seq_length":seq_length}

# --- Transformer (PyTorch) ---
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y): self.X = torch.tensor(X, dtype=torch.float32); self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, 1)
    def forward(self, x):
        x = self.embedding(x)
        for b in self.blocks: x = b(x)
        return self.fc(x[:, -1, :])

def transformer_model(trainval_df, test_df, logger=None):
    seq_length = SEQ_LENGTH_DEFAULT
    Xtr, ytr = create_sequences_for_window(trainval_df, seq_length, start_date=trainval_df['ds'].min(), end_date=train_cut_dt)
    Xv,  yv  = create_sequences_for_window(trainval_df, seq_length, start_date=train_cut_dt, end_date=val_end_dt)
    if len(Xtr)==0 and len(test_df)>0:
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"torch", "seq_length":seq_length}

    pin = (TORCH_DEVICE.type == "cuda")
    if len(Xtr)>0:
        train_loader = DataLoader(TimeSeriesDataset(Xtr, ytr), batch_size=NN_BATCH, shuffle=True, pin_memory=pin)

    model = TransformerModel(input_dim=(Xtr.shape[-1] if len(Xtr)>0 else 2), embed_dim=32, num_heads=4, ff_dim=128, num_layers=4).to(TORCH_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(TORCH_DEVICE.type=="cuda"))

    best_val = float("inf"); patience = EARLY_STOP_PATIENCE; noimp = 0; t0 = time.time(); best_state = None
    for epoch in range(100 if len(Xv)>0 else 60):
        model.train()
        if len(Xtr)>0:
            for xb, yb in train_loader:
                xb = xb.to(TORCH_DEVICE); yb = yb.to(TORCH_DEVICE)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(TORCH_DEVICE.type=="cuda")):
                    pred = model(xb).squeeze(); loss = loss_fn(pred, yb)
                scaler_amp.scale(loss).backward(); scaler_amp.step(opt); scaler_amp.update()

        if len(Xv) > 0:
            # validate
            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(TORCH_DEVICE.type=="cuda")):
                val_X = torch.tensor(Xv, dtype=torch.float32).to(TORCH_DEVICE)
                val_y = torch.tensor(yv, dtype=torch.float32).to(TORCH_DEVICE)
                vp = model(val_X).squeeze(); vloss = float(loss_fn(vp, val_y).item())
            if vloss + 1e-9 < best_val:
                best_val = vloss; noimp = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                noimp += 1
            if noimp >= patience: break
    train_seconds = time.time() - t0
    if best_state is not None: model.load_state_dict(best_state)

    if len(Xv) > 0:
        with torch.no_grad():
            vp_s = model(torch.tensor(Xv, dtype=torch.float32).to(TORCH_DEVICE)).detach().cpu().numpy().reshape(-1)
        vpred = inv_scale_safe(vp_s)
        vtrue = inv_scale_safe(yv)
    else:
        vpred, vtrue = np.array([]), np.array([])

    test_pred = roll_predict_nn_full(model, trainval_df, test_df, seq_length, framework='torch')

    if logger:
        logger.log_params("transformer", {"seq_length":seq_length, "embed_dim":32, "heads":4, "ff_dim":128, "layers":4,
                                          "epochs":"<=100(ES)_or_60", "batch_size":NN_BATCH,
                                          "train_seconds":float(train_seconds), "lib":"torch"})
        try: logger.save_model("transformer", model, kind="torch")
        except Exception: pass
    return test_pred, (vtrue, vpred), {"lib":"torch", "seq_length":seq_length}

import inspect
# import EarlyStopping safely once
# enforce compatibility with Darts Trainer
try:
    from pytorch_lightning.callbacks import EarlyStopping
except ImportError:
    from lightning.pytorch.callbacks import EarlyStopping


def _ts_from_scaled(df):
    d = df[['ds', 'y_scaled']].copy()
    zero_scaled_value = (0 - scaler.mean_[0]) / scaler.scale_[0]
    d['y_scaled'] = d['y_scaled'].ffill().fillna(zero_scaled_value)
    ts = TimeSeries.from_dataframe(d, time_col='ds', value_cols='y_scaled', freq='D')
    return ts.astype(np.float32)  # <- important: trainer precision '32-true' expects float32

def _pl_trainer_kwargs():
    return {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        # IMPORTANT: use a valid precision string and match the TimeSeries dtype
        "precision": "32-true",  # matches ts.astype(np.float32)
        # simple early stopping on training loss (we’re not passing a val series)
        "callbacks": [EarlyStopping(monitor="train_loss", patience=5, mode="min")],
    }



def nbeats_model(train_df, test_df, logger=None):
    if not (HAS_DARTS and HAS_NBEATS):
        print("[WARN] NBEATS unavailable → returning NaNs")
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"darts"}
    try:
        h = len(test_df)
        if h == 0:
            return np.array([]), (np.array([]), np.array([])), {"lib":"darts"}

        # float32 TimeSeries to match trainer precision
        ts_train = _ts_from_scaled(train_df).astype(np.float32)
        L = len(ts_train)
        in_len, out_len = _safe_chunk_lengths(L, h, desired_out=14, min_in=8, max_in=96)
        if in_len is None:
            print(f"[WARN] NBEATS: series too short (L={L}, h={h}) → skipping")
            return np.full(h, np.nan), (np.array([]), np.array([])), {"lib":"darts"}

        # lighter/faster NBEATS config
        m = NBEATSModel(
            input_chunk_length=in_len,
            output_chunk_length=out_len,
            n_epochs=60,           # lighter than default
            batch_size=64,
            num_blocks=1,          # fewer stacks
            num_layers=2,          # shallower
            layer_widths=128,      # narrower
            random_state=RANDOM_SEED,
            pl_trainer_kwargs=_pl_trainer_kwargs(),
        )

        # version-safe fit kwargs
        fit_kwargs = {}
        if "num_loader_workers" in inspect.signature(m.fit).parameters:
            fit_kwargs["num_loader_workers"] = 0  # avoid Windows worker overhead
        for k, v in [("force_reset", True), ("save_checkpoints", False)]:
            if k in inspect.signature(m.fit).parameters:
                fit_kwargs[k] = v

        try:
            m.fit(ts_train, **fit_kwargs)
        except TypeError:
            m.fit(ts_train)

        preds_scaled = m.predict(n=h)
        preds_unscaled = inv_scale_safe(preds_scaled.values())

        if logger:
            logger.log_params("nbeats", {
                "input_chunk_length": in_len, "output_chunk_length": out_len,
                "n_epochs": 60, "batch_size": 64,
                "num_blocks": 1, "num_layers": 2, "layer_widths": 128,
                "lib": "darts"
            })
            try: logger.save_model("nbeats", m, kind="darts")
            except Exception: pass

        return preds_unscaled, (np.array([]), np.array([])), {"lib":"darts"}

    except Exception as e:
        if logger: logger.log_exception("nbeats", e)
        print(f"[WARN] NBEATS failed ({e}) → NaNs")
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"darts"}


def nhits_model(train_df, test_df, logger=None):
    if not (HAS_DARTS and HAS_NHITS):
        print("[WARN] NHiTS unavailable → returning NaNs")
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"darts"}
    try:
        h = len(test_df)
        if h == 0:
            return np.array([]), (np.array([]), np.array([])), {"lib":"darts"}

        ts_train = _ts_from_scaled(train_df).astype(np.float32)  # match '32-true'
        L = len(ts_train)
        in_len, out_len = _safe_chunk_lengths(L, h, desired_out=14, min_in=8, max_in=128)
        if in_len is None:
            print(f"[WARN] NHiTS: series too short (L={L}, h={h}) → skipping")
            return np.full(h, np.nan), (np.array([]), np.array([])), {"lib":"darts"}

        m = NHiTSModel(
            input_chunk_length=in_len,
            output_chunk_length=out_len,
            n_epochs=60,
            batch_size=64,
            random_state=RANDOM_SEED,
            pl_trainer_kwargs=_pl_trainer_kwargs(),
        )

        fit_kwargs = {}
        if "num_loader_workers" in inspect.signature(m.fit).parameters:
            fit_kwargs["num_loader_workers"] = 0
        for k, v in [("force_reset", True), ("save_checkpoints", False)]:
            if k in inspect.signature(m.fit).parameters:
                fit_kwargs[k] = v

        try:
            m.fit(ts_train, **fit_kwargs)
        except TypeError:
            m.fit(ts_train)

        preds_scaled = m.predict(n=h)
        preds_unscaled = inv_scale_safe(preds_scaled.values())

        if logger:
            logger.log_params("nhits", {
                "input_chunk_length": in_len, "output_chunk_length": out_len,
                "n_epochs": 60, "batch_size": 64, "lib":"darts"
            })
            try: logger.save_model("nhits", m, kind="darts")
            except Exception: pass
        return preds_unscaled, (np.array([]), np.array([])), {"lib":"darts"}

    except Exception as e:
        if logger: logger.log_exception("nhits", e)
        print(f"[WARN] NHiTS failed ({e}) → NaNs")
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"darts"}

# --- PatchTST (HF) ---
def patchtst_model(trainval_df, test_df, logger=None):
    if not HAS_HF_PATCHTST:
        print("[WARN] PatchTST (HF) not available → NaNs")
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"hf-transformers"}

    # ---- split & observed, but use SCALED targets for stability ----
    trm = trainval_df[trainval_df['ds'] < train_cut_dt].copy()
    va  = trainval_df[(trainval_df['ds'] >= train_cut_dt) & (trainval_df['ds'] < val_end_dt)].copy()

    # use scaled labels (trained scaler from _rebuild_splits) and only observed rows
    obs = trm.loc[(trm.get('mask', 1).astype(bool)) & trm['y_scaled'].notna(), ['ds', 'y_scaled']].sort_values('ds')
    y_s = obs['y_scaled'].values.astype(np.float32)
    if y_s.size < 64:
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"hf-transformers"}

    def build_windows(yarr, in_len, out_len):
        X, Y = [], []
        for i in range(0, len(yarr) - in_len - out_len + 1):
            X.append(yarr[i:i + in_len])
            Y.append(yarr[i + in_len:i + in_len + out_len])
        if not X:
            return np.empty((0, in_len, 1), np.float32), np.empty((0, out_len, 1), np.float32)
        X = np.asarray(X, np.float32)[:, :, None]
        Y = np.asarray(Y, np.float32)[:, :, None]
        return X, Y

    L = len(y_s)
    in_len, out_len = _choose_window_lengths(L, out_len_default=14, min_in=32, max_in=192)
    Xall, Yall = build_windows(y_s, in_len, out_len)
    if Xall.shape[0] == 0:
        return np.full(len(test_df), np.nan), (np.array([]), np.array([])), {"lib":"hf-transformers"}

    vN = max(1, int(0.15 * Xall.shape[0]))
    if Xall.shape[0] <= vN:  # very short: fall back to train-only
        Xtr, Ytr, Xv, Yv = Xall, Yall, np.empty((0, in_len, 1), np.float32), np.empty((0, out_len, 1), np.float32)
    else:
        Xtr, Ytr = Xall[:-vN], Yall[:-vN]
        Xv,  Yv  = Xall[-vN:], Yall[-vN:]

    # ---- HF PatchTST: keep external scaling (scaling=None) ----
    cfg = PatchTSTConfig(
        num_input_channels=1,
        context_length=in_len,
        prediction_length=out_len,
        d_model=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        ffn_dim=256,
        loss="mse",
        scaling=None,              # we already scale with StandardScaler
        attention_dropout=0.0,
        ff_dropout=0.0,
        head_dropout=0.0,
    )
    model = PatchTSTForPrediction(cfg).to(TORCH_DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    bs = NN_BATCH
    N  = Xtr.shape[0]
    epochs = 100
    idx = np.arange(N)

    use_amp = (TORCH_DEVICE.type == "cuda")
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = float("inf")
    patience = EARLY_STOP_PATIENCE
    noimp = 0
    best_state = None
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        if N > 0:
            np.random.shuffle(idx)
            for s in range(0, N, bs):
                b = idx[s:s + bs]
                xb = torch.tensor(Xtr[b], device=TORCH_DEVICE)
                yb = torch.tensor(Ytr[b], device=TORCH_DEVICE)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(past_values=xb, future_values=yb)
                    loss = out.loss

                scaler_amp.scale(loss).backward()
                # gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(opt)
                scaler_amp.update()

        # ---- validation ----
        if Xv.shape[0] > 0:
            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                xv = torch.tensor(Xv, device=TORCH_DEVICE)
                yv = torch.tensor(Yv, device=TORCH_DEVICE)
                vl = float(model(past_values=xv, future_values=yv).loss.item())
            if vl + 1e-9 < best:
                best = vl
                noimp = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                noimp += 1
            if noimp >= patience:
                break

    train_seconds = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- iterative inference over test horizon (on scaled values) ----
    history = y_s.copy()
    preds_s = []
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        while len(preds_s) < len(test_df):
            past = history[-in_len:]
            xb = torch.tensor(past[None, :, None], device=TORCH_DEVICE)
            out = model(past_values=xb)
            step_pred = out.prediction_outputs.detach().cpu().numpy()[0, :, 0]
            preds_s.extend(step_pred.tolist())
            history = np.concatenate([history, step_pred.astype(np.float32)], axis=0)

    preds_s = np.asarray(preds_s[:len(test_df)], np.float32)
    # inverse-scale back to original units for downstream metrics/plots
    preds = inv_scale_safe(preds_s)

    # still return val pair structure (no explicit val preds here)
    vtrue = va['y'].to_numpy() if len(va) > 0 else np.array([])
    vpred = np.full_like(vtrue, np.nan, dtype=float)

    if logger:
        logger.log_params("patchtst", {
            "lib": "hf-transformers",
            "context_length": in_len,
            "prediction_length": out_len,
            "d_model": 64,
            "layers": 3,
            "heads": 4,
            "ffn_dim": 256,
            "epochs": "<=100(ES)",
            "batch_size": bs,
            "train_seconds": float(train_seconds),
            "amp": bool(use_amp),
            "grad_clip": 1.0,
            "external_scaling": True
        })
        try:
            logger.save_model("patchtst", model, kind="torch")
        except Exception:
            pass

    return preds.astype(float), (vtrue, vpred), {"lib": "hf-transformers"}


# -----------------------
# ENSEMBLE (inverse-MAE weights)
# -----------------------
def _mae(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = min(a.size, b.size)
    if m == 0: return np.inf
    a, b = a[-m:], b[-m:]
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask]))) if mask.any() else np.inf

def _left_pad_to_len(arr, T):
    arr = np.asarray(arr, float).ravel()
    if arr.size >= T: return arr[-T:]
    pad = np.full(T - arr.size, np.nan, float)
    return np.concatenate([pad, arr], axis=0)

def _weighted_blend(pred_list, weights, eps=1e-12):
    if not pred_list: return np.array([])
    P = np.vstack(pred_list); W = np.asarray(weights, float).reshape(-1, 1)
    valid = np.isfinite(P); Pz = np.where(valid, P, 0.0); Wt = np.where(valid, W, 0.0)
    denom = Wt.sum(axis=0, keepdims=True) + eps; Wt = Wt / denom
    return (Wt * Pz).sum(axis=0)
def _calibrate_on_val(vtrue, vpred):
    vtrue = np.asarray(vtrue, float).ravel()
    vpred = np.asarray(vpred, float).ravel()
    m = min(vtrue.size, vpred.size)
    if m == 0: return 0.0, 1.0
    y = vtrue[-m:]; x = vpred[-m:]
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 3: return 0.0, 1.0
    X = np.vstack([np.ones(mask.sum()), x[mask]]).T
    a, b = np.linalg.lstsq(X, y[mask], rcond=None)[0]  # y ≈ a + b*x
    return float(a), float(b)

def _simplex_project(w, eps=1e-12):
    w = np.maximum(np.asarray(w, float), 0)
    s = w.sum()
    return w / (s + eps) if s > 0 else np.full_like(w, 1.0/len(w))

def _stacking_weights(val_pairs, ridge=1e-4, temper=0.85):
    names = list(val_pairs.keys())
    if not names: return None
    # build aligned design matrix
    vt0, vp0 = val_pairs[names[0]]
    T = min(len(vt0), len(vp0))
    y = np.asarray(vt0[-T:], float)
    cols = [np.asarray(vp0[-T:], float)]
    for n in names[1:]:
        vt, vp = val_pairs[n]
        t = min(len(vt), len(vp), T)
        y  = y[-t:]
        cols = [c[-t:] for c in cols]
        cols.append(np.asarray(vp[-t:], float))
    P = np.column_stack(cols)
    mask = np.isfinite(y) & np.all(np.isfinite(P), axis=1)
    P = P[mask]; y = y[mask]
    if P.shape[0] < max(3, P.shape[1]):  # not enough points
        return None
    # ridge solution, then project to simplex
    G = P.T @ P + ridge * np.eye(P.shape[1])
    w = np.linalg.solve(G, P.T @ y)
    w = np.power(np.maximum(w, 0), temper)   # soften dominance
    return {n: float(x) for n, x in zip(names, _simplex_project(w))}

def ensemble_model(trainval_df, test_df, logger=None,
                   members=('nbeats','nhits','lgbm','patchtst','rf','transformer','lstm'),
                   base_funcs=None):
    if base_funcs is None: raise RuntimeError("ensemble_model needs base_funcs dict")
    T = len(test_df); preds_full = {}; val_pairs = {}; used = []; t_start = time.time()
    for name in members:
        if name not in base_funcs: print(f"[ENSEMBLE] Skip unknown member: {name}"); continue
        print(f"[ENSEMBLE] Fitting member: {name.upper()}")
        try:
            yhat, (vtrue, vpred), extras = base_funcs[name](trainval_df.copy(), test_df.copy(), logger)
        except Exception as e:
            if logger: logger.log_exception(name, e); continue
        yhat = np.asarray(yhat, float).ravel()
        if yhat.size == 0 or np.all(~np.isfinite(yhat)):
            print(f"[ENSEMBLE] {name} returned no predictions; skipped."); continue
        preds_full[name] = _left_pad_to_len(yhat, T)
        if vtrue is not None and vpred is not None and len(vtrue)>0 and len(vpred)>0:
            val_pairs[name] = (np.asarray(vtrue, float), np.asarray(vpred, float))
        used.append(name)

    if not preds_full:
        print("[ENSEMBLE] No base predictions → NaNs")
        return np.full(T, np.nan), (np.array([]), np.array([])), {"members": [], "lib":"core"}

    eps = 1e-6; weight_map = {}
    for name in used:
        if name in val_pairs:
            vt, vp = val_pairs[name]; w = 1.0 / (_mae(vt, vp) + eps)
        else:
            w = 0.25
        weight_map[name] = w
    wsum = sum(weight_map.values()) or float(len(weight_map))
    for k in weight_map: weight_map[k] = float(weight_map[k]) / wsum

    names = list(preds_full.keys()); matrix = [preds_full[n] for n in names]; weights = [weight_map[n] for n in names]
    ens_pred = _weighted_blend(matrix, weights)

    resids = []
    for n, pair in val_pairs.items():
        vt, vp = pair; m = min(vt.size, vp.size)
        if m:
            vt, vp = vt[-m:], vp[-m:]
            mask = np.isfinite(vt) & np.isfinite(vp)
            if mask.any(): resids.append(np.abs(vt[mask] - vp[mask]))
    qhat = float(np.quantile(np.concatenate(resids), 0.90)) if resids else np.nan

    meta = {"members": names, "weights": {n: float(w) for n, w in zip(names, weights)},
            "rule": "inverse-MAE (NaN-aware renorm)", "fit_seconds": float(time.time() - t_start),
            "conformal_q90_pooled": qhat if np.isfinite(qhat) else None, "lib":"core"}
    if logger:
        logger.log_params("ensemble", {"members": names, "weights": meta["weights"], "rule": meta["rule"], "lib":"core"})
        logger.log_metrics("ensemble_meta", {"fit_seconds": meta["fit_seconds"], "members": len(names),
                                             "conformal_q90_pooled": meta["conformal_q90_pooled"]})
    return ens_pred, (np.array([]), np.array([])), meta

# -----------------------
# Metrics
# -----------------------
def calculate_metrics(y_true, y_pred, y_train=None, seasonality=12, epsilon=1e-8):
    y_true = np.asarray(y_true, np.float64).ravel()
    y_pred = np.asarray(y_pred, np.float64).ravel()
    n = min(y_true.size, y_pred.size)
    keys = ["MSE","RMSE","MAE","MdAE","RMSLE","MAPE_%","sMAPE_%","R2","Corr","MBE","PBIAS_%",
            "NRMSE_mean","NRMSE_range","Theils_U1","Theils_U2","Directional_Accuracy_%","MASE"]
    if n == 0: return {k: float('nan') for k in keys}
    y_true = y_true[-n:]; y_pred = y_pred[-n:]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]; y_pred = y_pred[mask]
    if y_true.size == 0: return {k: float('nan') for k in keys}
    mse  = mean_squared_error(y_true, y_pred); rmse = np.sqrt(mse); mae = mean_absolute_error(y_true, y_pred)
    mdae = float(np.median(np.abs(y_pred - y_true)))
    mape  = float(np.mean(np.abs((y_true - y_pred) / (np.clip(np.abs(y_true), epsilon, None)))) * 100.0)
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100.0)
    rmsle = float(np.sqrt(np.mean((np.log1p(np.maximum(y_pred, 0)) - np.log1p(np.maximum(y_true, 0)))**2)))
    mbe   = float(np.mean(y_pred - y_true))
    pbias = float(100.0 * (np.sum(y_pred - y_true) / (np.sum(y_true) + epsilon)))
    r2    = float(r2_score(y_true, y_pred))
    corr  = float(np.corrcoef(y_true, y_pred)[0,1]) if y_true.size > 1 else np.nan
    nrmse_mean  = float(rmse / (np.mean(np.abs(y_true)) + epsilon))
    nrmse_range = float(rmse / (np.max(y_true) - np.min(y_true) + epsilon))
    u1 = float(rmse / (np.sqrt(np.mean(y_pred**2)) + np.sqrt(np.mean(y_true**2)) + epsilon))
    if y_true.size >= 2:
        denom_u2 = np.sqrt(np.mean(np.diff(y_true)**2)) + epsilon; u2 = float(rmse / denom_u2)
    else: u2 = np.nan
    if y_true.size >= 2 and y_pred.size >= 2:
        da = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100.0)
    else: da = np.nan
    mase = np.nan
    if y_train is not None:
        y_train = np.asarray(y_train, np.float64).ravel()
        if y_train.size > seasonality:
            denom = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality])) + epsilon
            mase = float(np.mean(np.abs(y_true - y_pred)) / denom)
    return {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "MdAE": float(mdae),
            "RMSLE": float(rmsle), "MAPE_%": float(mape), "sMAPE_%": float(smape),
            "R2": float(r2), "Corr": float(corr), "MBE": float(mbe), "PBIAS_%": float(pbias),
            "NRMSE_mean": float(nrmse_mean), "NRMSE_range": float(nrmse_range),
            "Theils_U1": float(u1), "Theils_U2": float(u2),
            "Directional_Accuracy_%": float(da), "MASE": float(mase)}

# -----------------------
# Main + CV driver
# -----------------------
def main():
    parser = argparse.ArgumentParser(description='Run specific models for price forecasting.')
    parser.add_argument('--model', type=str,
                        choices=['neuralprophet','tbats','lgbm','rf','lstm','nbeats','nhits','patchtst','transformer','ensemble','all'],
                        help='Specify which model to run.')
    parser.add_argument('--base_dir', type=str, default='experiments',
                        help='Directory to store experiment logs')
    parser.add_argument('--seasonality', type=str, default='auto',
                        help="Season length for MASE etc.; integer (e.g., 12/7/24) or 'auto'")
    # New: CV scheme controls
    parser.add_argument('--val_scheme', type=str, default='precut_60d', choices=['precut_60d','season_cv'],
                        help="Validation scheme: 'precut_60d' (original) or 'season_cv' (previous full seasons)")
    parser.add_argument('--season_start', type=str, default='09-15', help="Season start MM-DD (e.g., 09-01)")
    parser.add_argument('--season_end', type=str, default='11-15', help="Season end MM-DD (inclusive concept; we use exclusive end+1 day)")
    parser.add_argument('--fold_years', type=str, default='2022,2023', help="Comma-separated years used as validation seasons (e.g., '2022,2023')")
    parser.add_argument('--test_year', type=int, default=2024, help="Year to evaluate as the test season (e.g., 2024)")

    args = parser.parse_args()

    # Models dictionary
    models_to_run = {
        'neuralprophet': neuralprophet_model,
        'tbats': tbats_model,
        'lgbm': lgbm_model,
        'rf': random_forest_model,
        'lstm': lstm_model,
        'nbeats': nbeats_model,
        'nhits': nhits_model,
        'patchtst': patchtst_model,
        'transformer': transformer_model
    }
    def ensemble_wrapper(trv, te, log):
        members = ('nbeats','nhits','lgbm','rf','transformer','lstm')
        return ensemble_model(trv, te, log, members=members, base_funcs=models_to_run)
    models_to_run['ensemble'] = ensemble_wrapper

    # ------------------------------------
    # Logger setup (config filled later)
    # ------------------------------------
    darts_ver = getattr(darts, "__version__", "NA") if HAS_DARTS else "NA"
    print(f"[INFO] Using darts {darts_ver} | torch {torch.__version__}")
    if not HAS_PATCHTST_DARTS:
        print("[INFO] PatchTST (Darts) not available; using HF if present:", HAS_HF_PATCHTST)

    # ---- utility: infer seasonality for MASE
    def _infer_seasonality(y, max_period=400):
        if acf is None: return 1
        try:
            y = np.asarray(y, np.float64)
            if y.size < 8: return 1
            y = y - np.nanmean(y); y = np.nan_to_num(y)
            nlags = max(2, min(max_period, (len(y)//2) - 1))
            ac_vals = acf(y, nlags=nlags, fft=True)
            if ac_vals.shape[0] <= 1: return 1
            lag = 1 + int(np.argmax(ac_vals[1:]))
            return lag if lag > 1 and ac_vals[lag] > 0.1 else 1
        except Exception:
            return 1

    # Build a placeholder split so we can infer seasonality safely, then will be rebuilt as needed
    # Default original scheme:
    default_train_cut = pd.to_datetime('2024-09-15')
    default_val_end = default_train_cut + pd.Timedelta(days=VAL_DAYS)
    _rebuild_splits(default_train_cut, default_val_end, test_start_dt=default_val_end)

    if isinstance(args.seasonality, str) and args.seasonality.lower() == 'auto':
        seasonality_value = _infer_seasonality(TRAIN_OBS_Y); seasonality_mode = 'auto'
    else:
        try:
            seasonality_value = int(args.seasonality); seasonality_mode = 'fixed'
        except Exception:
            seasonality_value = 12; seasonality_mode = 'fixed'

    run_name = run_name = f"{DATASET_TAG}_{args.model or 'all'}_{args.val_scheme}"
    config = {"data_path": data_path,
              "val_scheme": args.val_scheme,
              "season_start": args.season_start, "season_end": args.season_end,
              "fold_years": args.fold_years, "test_year": args.test_year,
              "val_days": VAL_DAYS,
              "scaler": "StandardScaler",
              "torch_device": str(TORCH_DEVICE), "tf_gpu": bool(TF_HAS_GPU),
              "metrics_seasonality": seasonality_value, "metrics_seasonality_mode": seasonality_mode,
              "HAS_DARTS": bool(HAS_DARTS), "HAS_LGBM": bool(HAS_LGBM), "HAS_TBATS": bool(HAS_TBATS),
              "HAS_NP": bool(HAS_NP), "HAS_HF_PATCHTST": bool(HAS_HF_PATCHTST),
              "ONLINE_MODE": ONLINE_MODE}
    logger = ExperimentLogger(args.base_dir, run_name, config)

    print(f"Selected model: {args.model}")
    print(f"Env → DARTS={HAS_DARTS} | LGBM={HAS_LGBM} | TBATS={HAS_TBATS} | NP={HAS_NP} | HF_PatchTST={HAS_HF_PATCHTST}")
    print(f"Metrics seasonality -> {seasonality_value} (mode: {seasonality_mode})")
    print(f"ONLINE_MODE (teacher-forcing on test) -> {ONLINE_MODE}")

    # ---------- Split summary printing (single place) ----------
    if args.val_scheme == 'season_cv':
        # Parse folds and print a concise summary
        fold_years = [int(x.strip()) for x in str(args.fold_years).split(",") if str(x).strip()]
        folds = []
        ds_min = pd.to_datetime(full_data['ds'].min()) if 'ds' in full_data.columns else None
        for yr in fold_years:
            va_start, va_end_excl = _season_bounds(yr, args.season_start, args.season_end)
            tr_start = ds_min if ds_min is not None else va_start  # all history
            tr_end   = va_start - pd.Timedelta(days=1)
            va_end   = va_end_excl - pd.Timedelta(days=1)
            folds.append((tr_start, tr_end, va_start, va_end))
        test_start, test_end_excl = _season_bounds(args.test_year, args.season_start, args.season_end)

        for i, (tr_s, tr_e, va_s, va_e) in enumerate(folds, 1):
            print(f"Fold {i} → Train: [{tr_s.date()} … {tr_e.date()}] | Val: [{va_s.date()} … {va_e.date()}]")
        print(f"Final Test Season: [{test_start.date()} … {(test_end_excl - pd.Timedelta(days=1)).date()}]")
    else:
        print(f"Split → Train: < {default_train_cut.date()} | Val: [{default_train_cut.date()} … {(default_val_end - pd.Timedelta(days=1)).date()}] | Test: ≥ {default_val_end.date()}")

    # ------------------------------------
    # Helpers for running one model
    # ------------------------------------
    # Use globals for forced params so model funcs can see them
    global LGBM_FORCED_N_ESTIMATORS, RF_FORCED_MAX_DEPTH, RF_FORCED_MIN_LEAF

    def run_one(model_name, model_func):
        print(f"\n=== Running {model_name.upper()} ===")
        trv_df, te_df = dataset_for(model_name)

        start = time.time()
        try:
            out = model_func(trv_df, te_df, logger)
        except Exception as e:
            elapsed = time.time() - start
            logger.log_exception(model_name, e)
            # Always return 3 items so CV aggregation doesn't crash
            return (
                {"run_seconds": float(elapsed), "error": str(e)},
                {},
                (np.array([]), np.array([]))
            )
        elapsed = time.time() - start

        seq_length = None
        val_true = np.array([]); val_pred = np.array([]); extras = {}
        if isinstance(out, tuple):
            if len(out) == 3:
                yhat_full, (val_true, val_pred), third = out
                if isinstance(third, dict): extras = third
                else: seq_length = third
            elif len(out) == 2:
                yhat_full, (val_true, val_pred) = out
            else:
                yhat_full = out
        else:
            yhat_full = out

        yhat_full = np.asarray(yhat_full).reshape(-1)
        dates_all = te_df['ds'].values if 'ds' in te_df.columns else np.array([])
        y_all = te_df['y'].values if 'y' in te_df.columns else np.array([])
        obs_mask = np.isfinite(y_all)

        # Align prediction length
        if len(yhat_full) != len(te_df):
            if len(yhat_full) < len(te_df):
                pad = np.full(len(te_df)-len(yhat_full), np.nan); yhat_full = np.concatenate([pad, yhat_full])
            else:
                yhat_full = yhat_full[-len(te_df):]

        if len(te_df) > 0:
            print(f"{model_name}: test_len={len(te_df)}  pred_len={len(yhat_full)}  span={(pd.to_datetime(dates_all[0]).date() if len(dates_all)>0 else 'NA')}→{(pd.to_datetime(dates_all[-1]).date() if len(dates_all)>0 else 'NA')}  obs_days={int(obs_mask.sum())}")

            # Save FULL horizon
            full_pred_df = pd.DataFrame({"ds": pd.to_datetime(dates_all).astype(str), "y_true": y_all, "y_pred": yhat_full})
            full_path = logger.save_predictions_table(f"{model_name}_FULL", full_pred_df)

            # Save OBS-only
            y_true_obs = y_all[obs_mask]; y_pred_obs = yhat_full[obs_mask]; dates_obs = dates_all[obs_mask]
            obs_pred_df = pd.DataFrame({"ds": pd.to_datetime(dates_obs).astype(str), "y_true": y_true_obs, "y_pred": y_pred_obs})
            obs_path = logger.save_predictions_table(f"{model_name}_OBS", obs_pred_df)

            # Plots
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            _save_plot(dates_all, y_all, yhat_full, os.path.join(logger.figures_dir, f"{ts_tag}_{model_name}_FULL.png"),
                       f"Actual vs Predicted (FULL) - {model_name}")
            _save_plot(dates_obs, y_true_obs, y_pred_obs, os.path.join(logger.figures_dir, f"{ts_tag}_{model_name}_OBS.png"),
                       f"Actual vs Predicted (OBS) - {model_name}")

            # Metrics on OBS-only
            if y_true_obs.size > 0 and y_pred_obs.size > 0:
                metrics_full = calculate_metrics(y_true_obs, y_pred_obs, y_train=TRAIN_OBS_Y, seasonality=seasonality_value)
            else:
                metrics_full = {k: np.nan for k in ["MSE","RMSE","MAE","MdAE","RMSLE","MAPE_%","sMAPE_%","R2","Corr","MBE",
                                                    "PBIAS_%","NRMSE_mean","NRMSE_range","Theils_U1","Theils_U2",
                                                    "Directional_Accuracy_%","MASE"]}

            # Conformal intervals if val available
            lo = hi = np.full_like(y_pred_obs, np.nan, dtype=float); qhat = np.nan
            if val_true.size > 0 and val_pred.size > 0:
                lo, hi, qhat = conformal_intervals(val_true, val_pred, y_pred_obs, alpha=0.1)
                obs_ci_df = pd.DataFrame({"ds": pd.to_datetime(dates_obs).astype(str),
                                          "y_true": y_true_obs, "y_pred": y_pred_obs,
                                          "y_lo90": lo, "y_hi90": hi})
                logger.save_predictions_table(f"{model_name}_OBS_with_CI", obs_ci_df)
        else:
            # No test (e.g., CV fold), metrics on val if available
            metrics_full = {k: np.nan for k in ["MSE","RMSE","MAE","MdAE","RMSLE","MAPE_%","sMAPE_%","R2","Corr","MBE",
                                                "PBIAS_%","NRMSE_mean","NRMSE_range","Theils_U1","Theils_U2",
                                                "Directional_Accuracy_%","MASE"]}
            qhat = np.nan

        payload = {"run_seconds": float(elapsed), "metrics_seasonality": int(seasonality_value),
                   "metrics_seasonality_mode": str(seasonality_mode),
                   "conformal_q90": float(qhat) if np.isfinite(qhat) else np.nan,
                   "device": ("cuda" if torch.cuda.is_available() else "cpu"),
                   "scaler_mean": scaler_stats.get("mean", np.nan), "scaler_scale": scaler_stats.get("scale", np.nan)}
        if seq_length is not None: payload["seq_length"] = int(seq_length)
        if isinstance(extras, dict): payload.update(extras)
        payload.update(metrics_full)
        logger.log_metrics(model_name, payload)

        default_lib = ("hf-transformers" if model_name == "patchtst" and HAS_HF_PATCHTST else
                       "darts" if model_name in {"nbeats","nhits"} else
                       "sklearn" if model_name in {"lgbm","rf"} else
                       "tf" if model_name == "lstm" else
                       "torch" if model_name == "transformer" else
                       "tbats" if model_name == "tbats" else
                       "core")
        logger.log_time_row(model_name, elapsed, payload.get("device",""), default_lib)

        if len(te_df) > 0:
            print(f"{model_name.upper()} -> seconds: {elapsed:.2f}")
        return (payload, extras, (val_true, val_pred))

    # ------------------------------------
    # Run according to scheme
    # ------------------------------------
    if args.val_scheme == 'precut_60d':
        # Original behavior: Train < train_cut, Val = 60d after, Test ≥ val_end
        _rebuild_splits(default_train_cut, default_val_end, test_start_dt=default_val_end)
        to_run = list(models_to_run.items()) if (not args.model or args.model=='all') else [(args.model, models_to_run[args.model])]
        for name, fn in to_run:
            run_one(name, fn)

    else:
        # ----------- Season-aligned CV -----------
        fold_years = [int(x.strip()) for x in str(args.fold_years).split(",") if str(x).strip()]
        print(f"[CV] Season-aligned folds on years: {fold_years}  (season {args.season_start} → {args.season_end})")

        # Which models to CV?
        model_names = list(models_to_run.keys()) if (not args.model or args.model=='all') else [args.model]

        cv_store = {m: [] for m in model_names}
        for yr in fold_years:
            vstart, vend = _season_bounds(yr, args.season_start, args.season_end)
            # For CV fold, test is empty (or ignored), we rebuild splits; test_start at vend (no need for test_end)
            _rebuild_splits(cut_start_dt=vstart, cut_end_dt=vend, test_start_dt=vend, test_end_dt=vend)

            for name in model_names:
                print(f"[CV] {name.upper()} @ {yr}")
                payload, extras, (vtrue, vpred) = run_one(f"{name}_CV_fold{yr}", models_to_run[name])
                # Compute MAE on validation if available
                mae = np.inf
                if vtrue.size > 0 and vpred.size > 0:
                    vt, vp = np.asarray(vtrue, float), np.asarray(vpred, float)
                    m = min(vt.size, vp.size)
                    if m > 0:
                        vt, vp = vt[-m:], vp[-m:]
                        mask = np.isfinite(vt) & np.isfinite(vp)
                        if mask.any():
                            mae = float(np.mean(np.abs(vt[mask] - vp[mask])))
                cv_store[name].append({"year": yr, "MAE": mae, "extras": extras})

        # Aggregate hyper-params across folds
        agg_params = {}
        for name, results in cv_store.items():
            if not results: continue
            avg_mae = float(np.nanmean([r["MAE"] for r in results])) if any(np.isfinite(r["MAE"]) for r in results) else np.nan
            summary = {"avg_val_MAE": avg_mae, "folds": results}

            # LGBM aggregation
            if name == "lgbm":
                ns = [r["extras"].get("best_n") for r in results if r["extras"].get("best_n") is not None]
                if ns:
                    forced_n = int(np.round(np.mean(ns)))
                    summary["forced_n_estimators"] = forced_n
                    LGBM_FORCED_N_ESTIMATORS = forced_n  # set global

            # RF aggregation
            if name == "rf":
                cfgs = [(r["extras"].get("best_max_depth"), r["extras"].get("best_min_leaf")) for r in results]
                from collections import Counter
                common = Counter(cfgs)
                if common:
                    top_cfg, _ = common.most_common(1)[0]
                    # tiebreaker by mean MAE for tied configs
                    candidates = [cfg for cfg, cnt in common.items() if cnt == common.most_common(1)[0][1]]
                    if len(candidates) > 1:
                        best_cfg = None; best_mae = np.inf
                        for cfg in candidates:
                            maes = [r["MAE"] for r in results if (r["extras"].get("best_max_depth"), r["extras"].get("best_min_leaf")) == cfg]
                            m = np.nanmean(maes) if maes else np.inf
                            if m < best_mae: best_mae = m; best_cfg = cfg
                        top_cfg = best_cfg
                    summary["forced_max_depth"] = top_cfg[0]
                    summary["forced_min_leaf"] = top_cfg[1]
                    RF_FORCED_MAX_DEPTH = top_cfg[0]
                    RF_FORCED_MIN_LEAF = top_cfg[1]

            agg_params[name] = summary
            logger.log_params(f"{name}_cv_agg", summary)

        # ----------- Final refit on all data < TEST season start; test on full TEST season -----------
        test_start, test_end = _season_bounds(args.test_year, args.season_start, args.season_end)
        _rebuild_splits(cut_start_dt=test_start, cut_end_dt=test_start, test_start_dt=test_start, test_end_dt=test_end)
        print(f"\n[FINAL] Refit on ALL data < {test_start.date()} and test on season [{test_start.date()} … {(test_end-pd.Timedelta(days=1)).date()}]")
        print(f"[FINAL] Forced params → LGBM.n_estimators={LGBM_FORCED_N_ESTIMATORS} | RF.(max_depth,min_leaf)=({RF_FORCED_MAX_DEPTH},{RF_FORCED_MIN_LEAF})")

        to_run = list(models_to_run.items()) if (not args.model or args.model=='all') else [(args.model, models_to_run[args.model])]
        for name, fn in to_run:
            run_one(name, fn)

    logger.finalize()

if __name__ == "__main__":
    main()
