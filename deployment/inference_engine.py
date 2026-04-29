"""Inference engine: NHITS-based deployment for sale-period forecasting.

Loads a pre-trained NHiTS model from the experiments folder and generates
forecasts for the NEXT YEAR's sale-period dates only (e.g., Sep 2026 – Jan 2027
for Azadpur American, based on the defined sale periods in sale_periods.py).

CRITICAL: The saved NHITS models were trained on StandardScaler-transformed
('scaled') data, so inference must:
  1. Fit StandardScaler on historical observed data
  2. Build TimeSeries from scaled values
  3. Predict (predictions are in scaled space)
  4. Inverse-transform predictions back to Rs/kg
"""

import os
import re
import warnings
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

warnings.filterwarnings("ignore")

# Suppress pytorch-lightning and litmodels chatter
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

HAS_DARTS = False
try:
    import torch
    from darts.models import NHiTSModel
    from darts import TimeSeries
    from pytorch_lightning import Trainer
    HAS_DARTS = True
except Exception:
    pass


class silence:
    """Context manager to suppress stdout/stderr."""
    def __enter__(self):
        import sys
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc, tb):
        import sys
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def load_data(data_path: str):
    """Load dataset with standard column renaming."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    raw = pd.read_csv(data_path)
    raw = raw.rename(columns={"Date": "ds", "Avg Price (per kg)": "y", "Mask": "mask"})
    raw["y"] = pd.to_numeric(raw["y"], errors="coerce")
    raw["ds"] = pd.to_datetime(raw["ds"], format="mixed", dayfirst=False, errors="coerce")
    raw = raw.dropna(subset=["ds"]).reset_index(drop=True)
    raw["mask"] = raw["mask"].fillna(0).astype(int)
    raw = raw.sort_values("ds").reset_index(drop=True)
    return raw


class NHiTSEngine:
    """NHiTS engine: load pre-trained model, forecast next-year sale dates."""

    def __init__(self, data_path: str, model_path: str, market: str = None, variety: str = None, grade: str = None):
        self.data_path = data_path
        self.model_path = model_path
        self.market = market
        self.variety = variety
        self.grade = grade
        self.model = None
        self._cpu_trainer = None
        self.scaler = None

    def _load_model(self):
        """Load the saved NHiTS model with CPU-only trainer."""
        if not HAS_DARTS:
            raise RuntimeError(
                "darts library not installed. Install with: pip install 'u8darts[torch]'"
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"NHiTS model not found: {self.model_path}")

        torch.serialization.add_safe_globals([NHiTSModel])
        with silence():
            self.model = NHiTSModel.load(self.model_path, map_location=torch.device("cpu"))
            self._cpu_trainer = Trainer(
                accelerator="cpu",
                devices=1,
                enable_progress_bar=False,
                logger=False,
            )

    def _fit_scaler(self, hist_df: pd.DataFrame) -> pd.DataFrame:
        """Fit StandardScaler on observed historical points and add y_scaled column."""
        train_obs = hist_df[hist_df["mask"] == 1].copy()
        self.scaler = StandardScaler()
        if not train_obs.empty:
            self.scaler.fit(train_obs[["y"]])
        else:
            self.scaler.fit(np.array([[0.0]]))

        # Scale y values
        hist_df = hist_df.copy()
        hist_df["y_scaled"] = np.nan
        mask_obs = hist_df["mask"] == 1
        if mask_obs.any():
            hist_df.loc[mask_obs, "y_scaled"] = self.scaler.transform(
                hist_df.loc[mask_obs, ["y"]]
            ).flatten()
        return hist_df

    def _build_series(self, df: pd.DataFrame) -> "TimeSeries":
        """Build a Darts TimeSeries from historical observed data (<=2025).

        Steps:
        1. Filter to observed data <= 2025
        2. Fit StandardScaler and create y_scaled
        3. Aggregate duplicate dates by mean of scaled values
        4. Forward-fill missing daily dates
        5. Return TimeSeries with daily frequency (scaled values)
        """
        # Use only observed data up to end of 2025
        hist_df = df[(df["ds"] <= pd.Timestamp("2025-12-31")) & (df["mask"] == 1)].copy()
        if hist_df.empty:
            raise ValueError("No historical observed data found (mask==1, <=2025).")

        # Fit scaler on historical data and add y_scaled
        hist_df = self._fit_scaler(hist_df)

        # Aggregate duplicate dates by mean of SCALED price
        agg = hist_df.groupby("ds", as_index=False)["y_scaled"].mean()

        # Forward-fill missing daily dates
        agg = agg.set_index("ds").asfreq("D")
        agg["y_scaled"] = agg["y_scaled"].ffill()
        agg = agg.reset_index().rename(columns={"index": "ds"})
        agg = agg.dropna(subset=["y_scaled"])

        # Convert to float32 for model compatibility
        agg["y_scaled"] = agg["y_scaled"].astype(np.float32)

        series = TimeSeries.from_dataframe(agg, "ds", "y_scaled", fill_missing_dates=True, freq="D")
        return series

    def _get_sale_dates_2026(self) -> list:
        """Get the 2026/2027 sale-period dates for this market/variety/grade."""
        from deployment.sale_periods import get_sale_period, generate_sale_dates_2026

        if not all([self.market, self.variety, self.grade]):
            raise ValueError(
                "market, variety, and grade must be provided to determine sale period."
            )

        sale_info = get_sale_period(self.market, self.variety, self.grade)
        if not sale_info:
            raise ValueError(
                f"No sale period defined for {self.market}, {self.variety}, {self.grade}. "
                f"Check deployment/sale_periods.py."
            )

        sale_dates = generate_sale_dates_2026(sale_info["start"], sale_info["end"])
        if not sale_dates:
            raise ValueError(
                f"Could not generate sale dates for {self.market}, {self.variety}, {self.grade} "
                f"(period: {sale_info['start']} to {sale_info['end']})."
            )

        return sale_dates

    def forecast(self, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate forecast for the next year's sale-period dates.

        Parameters
        ----------
        horizon : int
            Number of sale-period dates to forecast (7, 15, or 30).

        Returns
        -------
        forecast_df : pd.DataFrame
            Columns: ds (date), forecast (price in Rs/kg). Only sale-period dates.
        hist_df : pd.DataFrame
            Last 90 days of historical observed data (mask==1, <=2025).
        """
        # ------------------------------------------------------------------
        # 1. Load data and model
        # ------------------------------------------------------------------
        df = load_data(self.data_path)
        self._load_model()

        # ------------------------------------------------------------------
        # 2. Build TimeSeries from SCALED historical observed data
        # ------------------------------------------------------------------
        with silence():
            series = self._build_series(df)

        # ------------------------------------------------------------------
        # 3. Get sale-period dates for 2026/2027
        # ------------------------------------------------------------------
        sale_dates = self._get_sale_dates_2026()
        sale_dates = sorted(sale_dates)
        target_dates = sale_dates[:horizon]

        if not target_dates:
            raise ValueError("No target sale-period dates available for forecasting.")

        # ------------------------------------------------------------------
        # 4. Predict with NHiTS model (predictions are in SCALED space)
        # ------------------------------------------------------------------
        # NHiTS generates a daily forecast starting from the day after the series end.
        # We predict enough days to cover all target dates.
        last_hist_date = series.end_time()
        days_ahead_list = [(pd.Timestamp(d) - last_hist_date).days for d in target_dates]
        max_days = max(days_ahead_list)

        if max_days <= 0:
            raise ValueError("Target dates are not after the last historical date.")

        with silence():
            fc = self.model.predict(
                n=max_days,
                series=series,
                trainer=self._cpu_trainer,
                verbose=False,
            )

        # Extract forecasts for specific target dates (still in SCALED space)
        preds_scaled = []
        for days_ahead in days_ahead_list:
            # fc starts at last_hist_date + 1 day
            # days_ahead = 1 corresponds to first forecast point
            idx = days_ahead - 1
            if 0 <= idx < len(fc):
                preds_scaled.append(fc.values().flatten()[idx])
            else:
                preds_scaled.append(np.nan)

        preds_scaled = np.array(preds_scaled, dtype=float)

        # ------------------------------------------------------------------
        # 5. Inverse-scale predictions back to Rs/kg
        # ------------------------------------------------------------------
        preds = self.scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        # ------------------------------------------------------------------
        # 6. Build output dataframes
        # ------------------------------------------------------------------
        forecast_df = pd.DataFrame({
            "ds": pd.to_datetime(target_dates),
            "forecast": preds,
        })

        # Historical data for charting (last 90 observed days, <=2025)
        hist_chart = df[
            (df["ds"] <= pd.Timestamp("2025-12-31")) & (df["mask"] == 1)
        ].sort_values("ds").tail(90).copy()

        # ------------------------------------------------------------------
        # 7. Post-process: clip negative/unreasonable forecasts
        # ------------------------------------------------------------------
        forecast_df = self._clip_forecasts(forecast_df, hist_chart)

        return forecast_df, hist_chart

    def _clip_forecasts(self, forecast_df: pd.DataFrame, hist_df: pd.DataFrame) -> pd.DataFrame:
        """Clip forecasts to reasonable bounds to avoid negative or absurd values.

        Some NHITS models can extrapolate into negative territory when forecasting
        far into the future. We clip to a floor based on historical minimum.
        """
        if forecast_df.empty or hist_df.empty:
            return forecast_df

        min_hist = hist_df["y"].min()
        # Floor: at least 0, and at least 20% of historical minimum
        # (prevents unreasonably low but positive values too)
        floor = max(0.0, min_hist * 0.2)
        forecast_df["forecast"] = forecast_df["forecast"].clip(lower=floor)

        return forecast_df

