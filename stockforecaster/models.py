"""Prediction models: statistical signal scorers + walk-forward ML ensemble.

Statistical layer (transparent, rule-based — what a TA would do by hand):
    momentum        : multi-horizon returns, RSI, MACD, +DI/-DI, Aroon
    trend           : price vs SMA20/50/200, ADX strength, Ichimoku cloud
    volume          : volume ratio, OBV slope, Chaikin MF, VZO
    mean_reversion  : z-score vs 20d, Bollinger %B, stochastic extremes
    volatility      : ATR% sweet spot, vol-of-vol contraction/expansion

Machine-learning layer (industry-standard cross-sectional approach):
    - RandomForest regressor + GradientBoosting regressor on next-day return
    - RandomForest classifier for P(next day up)
    - Temporal walk-forward split so OOS metrics are honest
    - Ensemble = mean of RF & GBM predictions, calibrated against the
      historical hit-rate of the models' own selected names

The two layers are blended by the ranker using config.WEIGHTS.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

from . import config
from .features import FEATURE_COLUMNS, build_dataset


def _clip100(x: float) -> float:
    return max(0.0, min(100.0, x))


# ---------------------------------------------------------------------------
# Statistical signals (per ticker, from its latest feature row)
# ---------------------------------------------------------------------------
def statistical_signals(row: pd.Series) -> dict[str, Any]:
    """Score one latest-feature row into 0-100 sub-scores with reasons."""
    reasons: list[str] = []
    scores: dict[str, float] = {}

    # --- Momentum -----------------------------------------------------------
    mom = 0.0
    for col, wgt in [("ret_5d", 0.25), ("ret_10d", 0.35), ("ret_20d", 0.40)]:
        v = row.get(col, np.nan)
        if pd.notna(v):
            mom += wgt * np.tanh(v * 8.0) * 100
    rsi14 = row.get("rsi14", 50.0)
    if rsi14 >= 70:
        mom += 30
        reasons.append(f"RSI14={rsi14:.0f} strong momentum (overbought)")
    elif rsi14 >= 55:
        mom += 40
        reasons.append(f"RSI14={rsi14:.0f} bullish momentum")
    elif rsi14 <= 35:
        mom -= 15
    macd_hist = row.get("macd_hist_pct", np.nan)
    if pd.notna(macd_hist) and macd_hist > 0:
        mom += 20
        reasons.append("MACD histogram positive")
    else:
        mom -= 10
    di_diff = row.get("plus_di_minus_minus", np.nan)
    if pd.notna(di_diff) and di_diff > 0:
        mom += 15
        reasons.append("+DI above -DI (bullish pressure)")
    scores["momentum"] = _clip100(50 + mom / 2)
    if scores["momentum"] >= 60 and mom > 50:
        reasons.append("Positive multi-horizon momentum")

    # --- Trend --------------------------------------------------------------
    trend = 0.0
    for col, label in [("dist_sma20_pct", "SMA20"), ("dist_sma50_pct", "SMA50"),
                       ("dist_sma200_pct", "SMA200")]:
        v = row.get(col, np.nan)
        if pd.notna(v):
            trend += 25 * np.tanh(v * 5.0)
            if v > 0.5:
                reasons.append(f"Above {label}")
            elif v < -0.5:
                reasons.append(f"Below {label} (resistance)")
    adx_ = row.get("adx", np.nan)
    if pd.notna(adx_):
        if adx_ >= config.ADX_TREND_STRENGTH and di_diff > 0:
            trend += 20
            reasons.append(f"Strong uptrend (ADX {adx_:.0f})")
        elif adx_ >= config.ADX_TREND_STRENGTH:
            trend -= 10
        elif adx_ < 20:
            trend -= 15
            reasons.append("Weak trend (ADX<20) — choppy")
    tk = row.get("tenkan_kijun", np.nan)
    if pd.notna(tk) and tk > 0:
        trend += 10
        reasons.append("Ichimoku Tenkan above Kijun")
    scores["trend"] = _clip100(50 + trend / 1.5)

    # --- Volume / money flow ------------------------------------------------
    vol = 0.0
    vr = row.get("vol_ratio", np.nan)
    if pd.notna(vr):
        if 1.2 <= vr <= 3.0:
            vol += 40
            reasons.append(f"Volume {vr:.1f}x average (accumulation)")
        elif vr > 3.0:
            vol += 15
        else:
            vol -= 15
    obv = row.get("obv_slope_rel", np.nan)
    if pd.notna(obv) and obv > 0:
        vol += 25
        reasons.append("OBV rising (buying pressure)")
    else:
        vol -= 15
    cmf = row.get("cmf20", np.nan)
    if pd.notna(cmf):
        vol += 20 * np.tanh(cmf * 6.0)
    vzo = row.get("vzo14", np.nan)
    if pd.notna(vzo):
        vol += 15 * np.tanh(vzo / 40.0)
    scores["volume"] = _clip100(50 + vol / 1.5)

    # --- Mean reversion (short-term next-day edge) ---------------------------
    mr = 0.0
    z = row.get("zscore_20", np.nan)
    if pd.notna(z):
        if -2.0 <= z <= -0.3:
            mr += 35
            reasons.append(f"Oversold z-score {z:.2f} (bounce setup)")
        elif 1.5 <= z <= 2.5:
            mr -= 20
    pct_b = row.get("bb_pct_b", np.nan)
    if pd.notna(pct_b):
        if pct_b <= 0.15:
            mr += 25
            reasons.append("At lower Bollinger Band")
        elif pct_b >= 0.9:
            mr -= 10
    sw = row.get("stoch_k", np.nan)
    if pd.notna(sw) and sw <= 20:
        mr += 15
    scores["mean_reversion"] = _clip100(50 + mr / 1.2)

    # --- Volatility suitability ---------------------------------------------
    atr_pct = row.get("atr_pct", np.nan)
    if pd.notna(atr_pct):
        if config.MIN_ATR_PCT * 100 <= atr_pct <= config.MAX_ATR_PCT * 100:
            scores["volatility"] = _clip100(50 + 20 * np.tanh((atr_pct - 4.0) / 3.0))
        else:
            scores["volatility"] = 20.0
            reasons.append("ATR% outside tradable band")
    else:
        scores["volatility"] = 50.0

    return {"scores": scores, "reasons": reasons}


# ---------------------------------------------------------------------------
# Machine-learning ensemble (walk-forward honest evaluation)
# ---------------------------------------------------------------------------
class ModelEnsemble:
    """RandomForest + GradientBoosting regressors + RF classifier, walk-forward split."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = {**config.ML, **(cfg or {})}
        self.regressors: list[Any] = []
        self.classifier: Optional[Any] = None
        self.feature_importances: dict[str, float] = {}
        self.metrics: dict[str, float] = {}
        self._calibration: dict[str, float] = {}

    def fit(self, dataset: pd.DataFrame) -> "ModelEnsemble":
        """Train on the in-sample portion; evaluate OOS on the last test span.

        Temporal split (no shuffling) mimics a real walk-forward and keeps the
        OOS metrics honest.
        """
        ds = dataset.dropna(subset=FEATURE_COLUMNS + ["target_ret", "target_up"]).copy()
        if len(ds) < 100:
            return self
        ds = ds.sort_values("date").reset_index(drop=True)
        split = int(len(ds) * (1 - self.cfg["test_size"]))
        train, test = ds.iloc[:split], ds.iloc[split:]
        Xtr, ytr_r, ytr_c = train[FEATURE_COLUMNS], train["target_ret"], train["target_up"]
        Xte, yte_r, yte_c = test[FEATURE_COLUMNS], test["target_ret"], test["target_up"]

        rf = RandomForestRegressor(n_estimators=self.cfg["n_estimators"],
                                   max_depth=self.cfg["max_depth"],
                                   min_samples_leaf=self.cfg["min_samples_leaf"],
                                   random_state=self.cfg["random_state"], n_jobs=-1)
        gb = GradientBoostingRegressor(n_estimators=min(self.cfg["n_estimators"], 200),
                                       max_depth=min(self.cfg["max_depth"], 4),
                                       loss="huber", random_state=self.cfg["random_state"])
        clf = RandomForestClassifier(n_estimators=self.cfg["n_estimators"],
                                     max_depth=self.cfg["max_depth"],
                                     min_samples_leaf=self.cfg["min_samples_leaf"],
                                     random_state=self.cfg["random_state"], n_jobs=-1)
        rf.fit(Xtr, ytr_r)
        gb.fit(Xtr, ytr_r)
        clf.fit(Xtr, ytr_c)

        pred_r = (rf.predict(Xte) + gb.predict(Xte)) / 2
        pred_c = clf.predict_proba(Xte)[:, 1]

        self.regressors = [rf, gb]
        self.classifier = clf
        self.feature_importances = {
            f: float((i_r + i_g) / 2)
            for f, i_r, i_g in zip(FEATURE_COLUMNS, rf.feature_importances_, gb.feature_importances_)
        }
        self.metrics = {
            "n_train": int(len(train)), "n_test": int(len(test)),
            "mae": float(mean_absolute_error(yte_r, pred_r)),
            "r2": float(r2_score(yte_r, pred_r)),
            "accuracy": float(accuracy_score(yte_c, (pred_c > 0.5).astype(int))),
            "base_rate_up": float(yte_c.mean()),
            "pred_mean_up": float((pred_c > 0.5).mean()),
        }

        # Calibration: how often did the top-quartile predicted names rise?
        thr = float(np.quantile(pred_r, 0.75))
        picked = test[pred_r >= thr]
        if len(picked):
            self._calibration = {
                "top_quartile_hit_rate": float((picked["target_up"] == 1).mean()),
                "top_quartile_expected_ret": float(picked["target_ret"].mean()),
                "threshold": thr,
            }
        return self

    def predict_latest(self, latest: pd.DataFrame) -> pd.DataFrame:
        """Score the most recent row of each ticker -> expected next-day move."""
        if not self.regressors:
            return pd.DataFrame(columns=["ticker", "pred_ret", "pred_up"])
        X = latest[FEATURE_COLUMNS].dropna()
        idx = latest.loc[X.index, "ticker"]
        rf, gb = self.regressors
        pred_ret = np.clip(
            (rf.predict(X) + gb.predict(X)) / 2,
            -self.cfg["clip_prediction"], self.cfg["clip_prediction"])
        pred_up = self.classifier.predict_proba(X)[:, 1]
        out = pd.DataFrame({"ticker": idx.values, "pred_ret": pred_ret, "pred_up": pred_up})
        hit = self._calibration.get("top_quartile_hit_rate")
        if hit is not None:
            out["pred_up"] = out["pred_up"] * 0.5 + hit * 0.5
        return out

    @property
    def is_fitted(self) -> bool:
        return bool(self.regressors)


# ---------------------------------------------------------------------------
# Historical base-rates (for calibration & explainability)
# ---------------------------------------------------------------------------
def historical_rates(dataset: pd.DataFrame) -> dict[str, float]:
    """P(next day up | momentum sign) from the dataset history."""
    ds = dataset.dropna(subset=["ret_5d", "target_up"])
    if len(ds) < 50:
        return {"base_rate_up": 0.5, "up_after_up5": 0.5, "up_after_down5": 0.5}
    base = float((ds["target_up"] == 1).mean())
    up5 = ds[ds["ret_5d"] > 0]
    dn5 = ds[ds["ret_5d"] < 0]
    return {
        "base_rate_up": base,
        "up_after_up5": float((up5["target_up"] == 1).mean()) if len(up5) else 0.5,
        "up_after_down5": float((dn5["target_up"] == 1).mean()) if len(dn5) else 0.5,
    }


def make_dataset(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return build_dataset(histories)


def fit_ensemble(histories: dict[str, pd.DataFrame]) -> tuple[ModelEnsemble, pd.DataFrame]:
    ds = build_dataset(histories)
    model = ModelEnsemble().fit(ds)
    return model, ds


