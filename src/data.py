"""Datos: carga, indicadores causales, escalamiento (solo con train) y split temporal.

Garantías anti-lookahead
------------------------
* Todos los indicadores usan ventanas *hacia atrás* (trailing) y se calculan DENTRO de cada
  día (groupby de día), de modo que nunca mezclan sesiones ni miran barras futuras.
* Como el split se hace en fronteras de día y los indicadores son locales al día, los
  indicadores de validación/test no dependen de datos de otro período.
* La media y la desviación del escalamiento se calculan SOLO con el período de train.
Estas propiedades se verifican automáticamente en tests/test_lookahead.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_ALIASES = {
    "timestamp": ["timestamp", "datetime", "date", "time", "fecha"],
    "open": ["open", "apertura"],
    "high": ["high", "maximo", "máximo"],
    "low": ["low", "minimo", "mínimo"],
    "close": ["close", "cierre", "adj close"],
    "volume": ["volume", "volumen"],
}


@dataclass
class SplitData:
    name: str
    features: np.ndarray          # (T, F) ya escalado, float32
    close: np.ndarray             # (T,)
    ts: np.ndarray                # (T,) datetime64
    day_id: np.ndarray            # (T,) int
    last_of_day: np.ndarray       # (T,) bool: última barra de cada sesión
    frame: pd.DataFrame = field(default=None, repr=False)  # OHLCV + indicadores SIN escalar (para EDA)

    def __len__(self) -> int:
        return len(self.close)


# --------------------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------------------
def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {}
    for std, aliases in _ALIASES.items():
        if std in df.columns:
            continue
        for a in aliases:
            if a in df.columns:
                rename[a] = std
                break
    df = df.rename(columns=rename)
    missing = [c for c in ["timestamp", "open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas {missing} en {path}. Columnas encontradas: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # conserva la hora local
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    df["volume"] = df["volume"].fillna(0.0)
    if (df["close"] <= 0).any():
        raise ValueError("Hay precios de cierre <= 0.")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def make_synthetic(
    n_days: int = 400,
    bars_per_day: int = 390,
    seed: int = 123,
    start_day: str = "2022-01-03",
    session_start: str = "09:30",
    phi: float = -0.05,
    base_vol_bps: float = 3.0,
) -> pd.DataFrame:
    """Serie intradía SINTÉTICA (volatilidad estocástica + patrón en U + AR(1) leve).

    Sirve únicamente para probar que el pipeline corre de punta a punta. Los resultados
    obtenidos con estos datos NO son evidencia sobre mercados reales.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start_day, periods=n_days)
    k = np.arange(bars_per_day)
    shape = 1 + 0.8 * (np.exp(-k / 20) + np.exp(-(bars_per_day - 1 - k) / 30))
    shape = shape / shape.mean()
    h, m = (int(x) for x in session_start.split(":"))
    logvol, price = 0.0, 100.0
    frames = []
    for day in days:
        logvol = 0.95 * logvol + 0.25 * rng.normal()
        sig = base_vol_bps * 1e-4 * np.exp(logvol)
        eps = rng.normal(size=bars_per_day)
        drift = rng.normal(0.0, 5e-3) / bars_per_day
        r = np.empty(bars_per_day)
        prev = 0.0
        for i in range(bars_per_day):
            r[i] = phi * prev + sig * shape[i] * eps[i] + drift
            prev = r[i]
        price *= np.exp(rng.normal(0.0, 2e-3))  # gap overnight
        close = price * np.exp(np.cumsum(r))
        open_ = np.concatenate([[price], close[:-1]])
        noise_h = np.abs(rng.normal(0, 0.5 * sig, bars_per_day))
        noise_l = np.abs(rng.normal(0, 0.5 * sig, bars_per_day))
        high = np.maximum(open_, close) * (1 + noise_h)
        low = np.minimum(open_, close) * (1 - noise_l)
        vol = 1000 * shape * np.exp(0.3 * rng.normal(size=bars_per_day) + 0.5 * logvol)
        ts = day + pd.Timedelta(hours=h, minutes=m) + pd.to_timedelta(k, unit="m")
        frames.append(pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                                    "close": close, "volume": vol.round()}))
        price = close[-1]
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Indicadores causales
# --------------------------------------------------------------------------------------
def feature_names(fcfg: dict) -> list[str]:
    return [f"ret_{l}" for l in fcfg["ret_lags"]] + ["dist_sma", "vol", "rsi", "vol_z", "tod"]


NO_SCALE = {"rsi", "tod"}


def _roll(s: pd.Series, day: pd.Series, window: int, fn: str, min_periods: int) -> pd.Series:
    res = getattr(s.groupby(day).rolling(window, min_periods=min_periods), fn)()
    return res.droplevel(0).sort_index()


def compute_features(df: pd.DataFrame, fcfg: dict, session_start: str, session_minutes: int) -> pd.DataFrame:
    """Devuelve df + day_id + indicadores SIN escalar. Todo trailing y local al día."""
    out = df.copy().reset_index(drop=True)
    out["day_id"] = out["timestamp"].dt.normalize().factorize()[0]
    d = out["day_id"]
    close = out["close"].astype(float)
    logc = np.log(close)

    for lag in fcfg["ret_lags"]:
        out[f"ret_{lag}"] = logc.groupby(d).diff(lag)
    out["dist_sma"] = np.log(close / _roll(close, d, fcfg["sma_window"], "mean", 1))

    ret1 = logc.groupby(d).diff(1).fillna(0.0)
    out["vol"] = _roll(ret1, d, fcfg["vol_window"], "std", 2)

    delta = close.groupby(d).diff().fillna(0.0)
    ag = _roll(delta.clip(lower=0), d, fcfg["rsi_window"], "mean", 1)
    al = _roll((-delta).clip(lower=0), d, fcfg["rsi_window"], "mean", 1)
    denom = (ag + al).to_numpy()
    rsi = np.where(denom > 0, 100.0 * ag.to_numpy() / np.where(denom > 0, denom, 1.0), 50.0)
    out["rsi"] = (rsi - 50.0) / 50.0  # en [-1, 1]

    v = out["volume"].astype(float)
    m = _roll(v, d, fcfg["volume_window"], "mean", 5)
    s = _roll(v, d, fcfg["volume_window"], "std", 5)
    out["vol_z"] = (v - m) / s.where(s > 0)

    h, mi = (int(x) for x in session_start.split(":"))
    minutes = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    frac = (minutes - (h * 60 + mi)) / float(session_minutes)
    out["tod"] = (2 * frac - 1).clip(-1, 1)

    cols = feature_names(fcfg)
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["last_of_day"] = (d != d.shift(-1)).to_numpy()
    return out


# --------------------------------------------------------------------------------------
# Escalamiento (solo train) y split
# --------------------------------------------------------------------------------------
def fit_scaler(frame: pd.DataFrame, cols: list[str]) -> dict:
    mean, std = {}, {}
    for c in cols:
        if c in NO_SCALE:
            mean[c], std[c] = 0.0, 1.0
        else:
            mean[c] = float(frame[c].mean())
            std[c] = float(frame[c].std() or 1.0)
    return {"cols": cols, "mean": mean, "std": std}


def apply_scaler(frame: pd.DataFrame, scaler: dict, clip: float) -> np.ndarray:
    x = np.stack([(frame[c].to_numpy() - scaler["mean"][c]) / scaler["std"][c] for c in scaler["cols"]], axis=1)
    return np.clip(x, -clip, clip).astype(np.float32)


def temporal_split(frame: pd.DataFrame, fr: dict) -> dict[str, pd.DataFrame]:
    n_days = int(frame["day_id"].max()) + 1
    c1 = int(n_days * fr["train"])
    c2 = int(n_days * (fr["train"] + fr["val"]))
    if c1 < 5 or c2 - c1 < 2 or n_days - c2 < 2:
        raise ValueError(f"Muy pocos días ({n_days}) para dividir en train/val/test.")
    d = frame["day_id"]
    return {
        "train": frame[d < c1].reset_index(drop=True),
        "val": frame[(d >= c1) & (d < c2)].reset_index(drop=True),
        "test": frame[d >= c2].reset_index(drop=True),
    }


def _to_split(name: str, frame: pd.DataFrame, scaler: dict, clip: float) -> SplitData:
    return SplitData(
        name=name,
        features=apply_scaler(frame, scaler, clip),
        close=frame["close"].to_numpy(dtype=np.float64),
        ts=frame["timestamp"].to_numpy(),
        day_id=frame["day_id"].to_numpy(),
        last_of_day=frame["last_of_day"].to_numpy(),
        frame=frame,
    )


def prepare_data(cfg: dict, scaler: dict | None = None) -> dict:
    """Carga/genera datos, calcula indicadores, divide y escala. Devuelve splits + scaler."""
    dcfg = cfg["data"]
    if dcfg.get("path"):
        raw = load_ohlcv(dcfg["path"])
    else:
        s = dcfg["synthetic"]
        raw = make_synthetic(s["n_days"], s["bars_per_day"], s["seed"], session_start=dcfg["session_start"])
    per_day = raw.groupby(raw["timestamp"].dt.normalize()).size().median()
    if per_day < 5:
        raise ValueError(
            f"El dataset tiene ~{per_day:.0f} barra(s) por día. Scalping requiere datos INTRADÍA "
            "(1, 5, 15 min). Con datos diarios usa swing/posicional o pide el dataset intradía.")
    frame = compute_features(raw, cfg["features"], dcfg["session_start"], dcfg["session_minutes"])
    parts = temporal_split(frame, dcfg["split"])
    cols = feature_names(cfg["features"])
    if scaler is None:
        scaler = fit_scaler(parts["train"], cols)  # SOLO train
    clip = cfg["features"]["clip"]
    out = {k: _to_split(k, v, scaler, clip) for k, v in parts.items()}
    out["scaler"] = scaler
    out["feature_names"] = cols
    return out


def save_scaler(scaler: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2)


def load_scaler(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Genera un CSV sintético de prueba.")
    ap.add_argument("--make-synthetic", action="store_true")
    ap.add_argument("--out", default="data/synthetic_1min.csv")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    if a.make_synthetic:
        make_synthetic(a.days, 390, a.seed).to_csv(a.out, index=False)
        print("escrito", a.out)
