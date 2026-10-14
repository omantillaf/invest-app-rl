"""Compara versiones de la función de recompensa (iteración de diseño) en VALIDACIÓN.

    python -m src.experiments --rewards v1_return v2_turnover v3_risk --seeds 0 1 2

Entrena cada (recompensa, semilla), evalúa el mejor checkpoint en el período de validación
y escribe reports/reward_comparison_val.{csv,md,png}. NO toca el período test.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import clone, eval_cfg, load_config
from .data import prepare_data
from .evaluate import evaluate_all, load_agent, summarize
from .metrics import compute_metrics, run_episode
from .environment import TradingEnv
from .train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--rewards", nargs="+", default=["v1_return", "v2_turnover", "v3_risk"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--random-seeds", type=int, default=10)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()

    base = load_config(a.config, a.set)
    os.makedirs(a.reports_dir, exist_ok=True)
    bpy = base["evaluation"]["bars_per_year"]
    data = prepare_data(base)            # mismos datos y mismo scaler para todos los runs
    val = data["val"]

    rows, curves = [], {}
    for name in a.rewards:
        for seed in a.seeds:
            cfg = clone(base)
            cfg["reward"]["name"], cfg["seed"] = name, seed
            run_dir = os.path.join(a.runs_dir, f"{name}_s{seed}")
            print(f"\n=== {name} | seed {seed} ===")
            info = train(cfg, run_dir, data=data)
            agent = load_agent(cfg, data, os.path.join(run_dir, "best.pt"))
            rec = run_episode(TradingEnv(val, eval_cfg(cfg)), agent)
            m = compute_metrics(rec, bpy)
            m.update({"recompensa": name, "seed": seed, "best_step": info["best_step"]})
            rows.append(m)
            curves.setdefault(name, []).append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.reports_dir, "reward_comparison_val_runs.csv"), index=False)
    keys = ["retorno_acumulado", "max_drawdown", "costos_total", "n_cambios_posicion",
            "sharpe_anualizado", "pct_tiempo_invertido"]
    agg = df.groupby("recompensa")[keys].agg(["mean", "std"])
    agg.to_csv(os.path.join(a.reports_dir, "reward_comparison_val.csv"))

    # Referencias en validación (baselines bajo las mismas condiciones)
    ref, ref_sd = summarize(evaluate_all(base, val, None, n_random=a.random_seeds), bpy)
    lines = ["| Estrategia (VALIDACIÓN) | Retorno acum. | Máx. drawdown | Costos ($) | Nº cambios | Sharpe | % invertido |",
             "|---|---|---|---|---|---|---|"]
    for name, g in df.groupby("recompensa"):
        f = lambda k, p=False: (f"{100 * g[k].mean():.2f}% ± {100 * g[k].std():.2f}" if p
                                else f"{g[k].mean():,.2f} ± {g[k].std():,.2f}")
        lines.append(f"| RL {name} (n={len(g)}) | {f('retorno_acumulado', True)} | {f('max_drawdown', True)} | "
                     f"{g['costos_total'].mean():,.0f} | {g['n_cambios_posicion'].mean():,.0f} | "
                     f"{g['sharpe_anualizado'].mean():.2f} | {100 * g['pct_tiempo_invertido'].mean():.0f}% |")
    for name in ref.index:
        r = ref.loc[name]
        lines.append(f"| {name} | {100 * r['retorno_acumulado']:.2f}% | {100 * r['max_drawdown']:.2f}% | "
                     f"{r['costos_total']:,.0f} | {r['n_cambios_posicion']:,.0f} | {r['sharpe_anualizado']:.2f} | "
                     f"{100 * r['pct_tiempo_invertido']:.0f}% |")
    md = "\n".join(lines)
    with open(os.path.join(a.reports_dir, "reward_comparison_val.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print("\n" + md)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, recs in curves.items():
        ax.plot(recs[0]["ts"], recs[0]["equity"], label=f"RL {name} (seed {a.seeds[0]})")
    ax.set_title("Comparación de recompensas — validación")
    ax.set_ylabel("Valor del portafolio ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(a.reports_dir, "reward_comparison_val.png"), dpi=130)
    print("Reportes en", a.reports_dir)


if __name__ == "__main__":
    main()
