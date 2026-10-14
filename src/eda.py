"""Análisis exploratorio. Usa SOLO train + validación (el held-out test no se toca).

    python -m src.eda --config configs/default.yaml
Genera figuras en reports/figures y un resumen numérico en reports/eda_summary.md
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .data import feature_names, prepare_data


def _daily(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby("day_id")
    return pd.DataFrame({
        "date": g["timestamp"].first().dt.normalize(),
        "close": g["close"].last(),
        "vol": g["ret_1"].std() * np.sqrt(g.size()),     # volatilidad realizada diaria
        "volume": g["volume"].sum(),
    })


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=130)
    plt.close(fig)


def run_eda(cfg: dict, out_dir: str = "reports/figures") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    data = prepare_data(cfg)
    tr, va = data["train"].frame, data["val"].frame
    fr = pd.concat([tr, va], ignore_index=True)           # test excluido
    cols = feature_names(cfg["features"])
    r = fr["ret_1"]
    daily = _daily(fr)
    val_start = va["timestamp"].iloc[0]
    s = {}

    # 1) Evolución del precio
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily["date"], daily["close"], lw=1)
    ax.axvline(val_start, color="r", ls="--", label="inicio validación")
    ax.set_title("Evolución del precio (cierre diario) — train + validación")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "01_precio.png")

    # 2) Distribución de retornos
    nz = r[r != 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(nz, bins=200, density=True, alpha=0.7, log=True, label="retornos 1 barra")
    xs = np.linspace(nz.min(), nz.max(), 400)
    ax.plot(xs, np.exp(-0.5 * ((xs - nz.mean()) / nz.std()) ** 2) / (nz.std() * np.sqrt(2 * np.pi)),
            "r", label="normal equivalente")
    ax.set_title("Distribución de retornos logarítmicos (escala log)")
    ax.legend()
    _save(fig, out_dir, "02_retornos.png")
    s.update({"ret_media": r.mean(), "ret_std": r.std(), "ret_skew": r.skew(), "ret_kurtosis_exceso": r.kurt(),
              "frac_retornos_cero": float((r == 0).mean())})

    # 3) Regímenes de volatilidad
    q1, q2 = daily["vol"].quantile([1 / 3, 2 / 3])
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(daily["date"], daily["vol"], lw=0.9)
    ax.axhline(q1, color="g", ls=":", label="baja/media (tercil 1)")
    ax.axhline(q2, color="r", ls=":", label="media/alta (tercil 2)")
    ax.axvline(val_start, color="k", ls="--")
    ax.set_title("Volatilidad realizada diaria (regímenes)")
    ax.legend()
    _save(fig, out_dir, "03_volatilidad.png")
    s.update({"vol_diaria_media": daily["vol"].mean(), "vol_tercil_bajo": q1, "vol_tercil_alto": q2})

    # 4) Perfil intradía
    mod = fr["timestamp"].dt.hour * 60 + fr["timestamp"].dt.minute
    prof = pd.DataFrame({"m": mod, "absret": r.abs(), "volume": fr["volume"]}).groupby("m").mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    a1.plot(prof.index, prof["absret"])
    a1.set_title("|retorno| medio por minuto del día")
    a2.plot(prof.index, prof["volume"])
    a2.set_title("Volumen medio por minuto del día")
    _save(fig, out_dir, "04_perfil_intradia.png")

    # 5) Caídas del precio
    peak = daily["close"].cummax()
    dd = daily["close"] / peak - 1
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(daily["date"], dd, 0, color="C3", alpha=0.5)
    ax.set_title("Drawdown del precio (comprar y mantener, cierres diarios)")
    _save(fig, out_dir, "05_drawdown_precio.png")
    s["max_drawdown_precio"] = float(dd.min())

    # 6) Autocorrelación de retornos
    x = r.to_numpy()
    acf = [np.corrcoef(x[:-k], x[k:])[0, 1] for k in range(1, 31)]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(1, 31), acf)
    ax.axhline(1.96 / np.sqrt(len(x)), color="r", ls=":")
    ax.axhline(-1.96 / np.sqrt(len(x)), color="r", ls=":")
    ax.set_title("Autocorrelación de retornos de 1 barra")
    _save(fig, out_dir, "06_acf_retornos.png")
    s["acf_lag1"] = acf[0]

    # 7) Indicadores: histogramas y correlación (train)
    fig, axs = plt.subplots(2, (len(cols) + 1) // 2, figsize=(12, 5))
    for ax, c in zip(axs.ravel(), cols):
        ax.hist(tr[c], bins=80)
        ax.set_title(c, fontsize=9)
    _save(fig, out_dir, "07_indicadores_hist.png")
    cm = tr[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)), cols, rotation=60)
    ax.set_yticks(range(len(cols)), cols)
    fig.colorbar(im)
    ax.set_title("Correlación entre indicadores (train)")
    _save(fig, out_dir, "08_indicadores_corr.png")

    # 8) Relación indicador -> retorno de la SIGUIENTE barra (solo análisis, en train)
    nxt = tr["close"].shift(-1) / tr["close"] - 1
    same_day = (~tr["last_of_day"]).to_numpy()
    corr_next = {c: float(np.corrcoef(tr[c][same_day][:-1], nxt[same_day][:-1])[0, 1]) for c in cols}
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(list(corr_next), list(corr_next.values()))
    ax.set_title("Correlación indicador(t) vs retorno(t→t+1) — train")
    plt.setp(ax.get_xticklabels(), rotation=45)
    _save(fig, out_dir, "09_indicador_vs_retorno_siguiente.png")
    s["corr_indicador_retorno_siguiente"] = corr_next

    s.update({"n_barras": len(fr), "n_dias": int(fr["day_id"].nunique()),
              "rango": f"{fr['timestamp'].iloc[0]} → {fr['timestamp'].iloc[-1]}"})
    with open(os.path.join(os.path.dirname(out_dir.rstrip('/')) or ".", "eda_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Resumen EDA (train + validación)\n\n")
        for k, v in s.items():
            f.write(f"- **{k}**: {v}\n")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    summary = run_eda(load_config(a.config, a.set), a.out)
    for k, v in summary.items():
        print(k, ":", v)
