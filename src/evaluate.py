"""Evaluación del agente y los baselines sobre un período (val o test) con las mismas condiciones.

    python -m src.evaluate --run runs/v3_risk_s0 --split val
    python -m src.evaluate --run runs/v3_risk_s0 --split test --final     # held-out, UNA sola vez

El período test exige --final para evitar usarlo por accidente al elegir hiperparámetros.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .agent import DQNAgent
from .baselines import BuyAndHold, MovingAverageCross, RandomAgent
from .config import clone, eval_cfg, load_config, set_seed
from .data import load_scaler, prepare_data
from .environment import TradingEnv
from .metrics import compute_metrics, run_episode

RANDOM = "Aleatorio"
BH = "Comprar y mantener"
BH_INTRA = "B&H intradía"
MA = "Cruce de medias"


def evaluate_all(cfg: dict, split, agent=None, agent_name: str = "Agente RL", n_random: int | None = None) -> dict:
    """Devuelve {nombre: registro}; el aleatorio es una lista de registros (varias semillas)."""
    cfg = eval_cfg(cfg)  # sin regla de ruina: todos recorren el período completo
    n_random = cfg["evaluation"]["random_seeds"] if n_random is None else n_random
    env = TradingEnv(split, cfg)
    n_act = env.action_space.n
    recs: dict = {}
    if agent is not None:
        recs[agent_name] = run_episode(env, agent)
    recs[RANDOM] = [run_episode(env, RandomAgent(n_act, seed=1000 + i)) for i in range(n_random)]
    recs[BH] = run_episode(TradingEnv(split, cfg, flat_at_close=False), BuyAndHold(n_act))
    recs[BH_INTRA] = run_episode(env, BuyAndHold(n_act))
    ev = cfg["evaluation"]
    recs[MA] = run_episode(env, MovingAverageCross(n_act, ev["ma_fast"], ev["ma_slow"]))
    return recs


def summarize(recs: dict, bars_per_year: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla de métricas (media) y desviaciones (solo para el aleatorio)."""
    rows, stds = {}, {}
    for name, rec in recs.items():
        if isinstance(rec, list):
            m = pd.DataFrame([compute_metrics(r, bars_per_year) for r in rec])
            rows[f"{name} (media de {len(rec)})"] = m.mean()
            stds[f"{name} (media de {len(rec)})"] = m.std()
        else:
            rows[name] = pd.Series(compute_metrics(rec, bars_per_year))
    return pd.DataFrame(rows).T, pd.DataFrame(stds).T


def to_markdown(df: pd.DataFrame, stds: pd.DataFrame | None = None) -> str:
    cols = [("retorno_acumulado", "Retorno acum.", "pct"), ("max_drawdown", "Máx. drawdown", "pct"),
            ("costos_total", "Costos ($)", "money"), ("costos_pct_capital", "Costos (% cap. inicial)", "pct"),
            ("n_cambios_posicion", "Nº cambios pos.", "int"), ("sharpe_anualizado", "Sharpe anual.", "f2"),
            ("pct_tiempo_invertido", "% tiempo invertido", "pct")]

    def fmt(v, kind):
        return {"pct": f"{100 * v:.2f}%", "money": f"{v:,.0f}", "int": f"{v:,.0f}", "f2": f"{v:.2f}"}[kind]

    lines = ["| Estrategia | " + " | ".join(c[1] for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name in df.index:
        cells = []
        for key, _, kind in cols:
            s = fmt(df.loc[name, key], kind)
            if stds is not None and name in stds.index:
                s += f" ± {fmt(stds.loc[name, key], kind)}"
            cells.append(s)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------------------
def plot_equity(recs: dict, path: str, title: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, rec in recs.items():
        if isinstance(rec, list):
            eq = np.stack([r["equity"] for r in rec])
            x = rec[0]["ts"]
            ax.plot(x, np.median(eq, 0), color="gray", lw=1, label=f"{name} (mediana)")
            ax.fill_between(x, np.percentile(eq, 10, 0), np.percentile(eq, 90, 0), color="gray", alpha=0.2)
        else:
            ax.plot(rec["ts"], rec["equity"], lw=1.6 if "RL" in name else 1.1, label=name)
    ax.set_title(title)
    ax.set_ylabel("Valor del portafolio ($)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_positions(rec: dict, path: str, bars: int = 1170):
    n = min(bars, len(rec["close"]))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(rec["ts"][:n], rec["close"][:n], lw=0.8)
    a1.set_ylabel("Precio")
    a2.step(rec["ts"][:n], rec["position"][:n], where="post")
    a2.set_ylabel("Posición")
    a2.set_ylim(-0.05, 1.05)
    a1.set_title(f"Posiciones del agente (primeras {n} barras del período)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tod_profile(rec: dict, path: str):
    ts = pd.to_datetime(rec["ts"][1:])
    df = pd.DataFrame({"min": ts.hour * 60 + ts.minute, "pos": rec["position"][1:], "turn": rec["turnover"]})
    df["bucket"] = (df["min"] // 30) * 30
    g = df.groupby("bucket").agg(pos=("pos", "mean"), trades=("turn", lambda s: (s > 0).mean()))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(g.index, g["pos"], width=25, label="Posición media")
    ax.plot(g.index, g["trades"], "o-", color="C3", label="Frecuencia de cambios")
    ax.set_xlabel("Minuto del día (bucket de 30 min)")
    ax.legend()
    ax.set_title("Comportamiento del agente según la hora del día")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def load_run(run_dir: str, overrides: list[str] | None = None):
    cfg = load_config(os.path.join(run_dir, "config.yaml"), overrides)
    scaler = load_scaler(os.path.join(run_dir, "scaler.json"))
    data = prepare_data(cfg, scaler=scaler)
    return cfg, data


def load_agent(cfg: dict, data: dict, ckpt: str) -> DQNAgent:
    env = TradingEnv(data["train"], cfg)
    return DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg, cfg["seed"]).load(ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Carpeta del run (con config.yaml, scaler.json, best.pt)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="Requerido para evaluar el held-out (test)")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="Sobrescribe el costo total por turnover (fee=X, slippage=0). Útil para sensibilidad.")
    ap.add_argument("--random-seeds", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], help="override de config, p. ej. env.execution_lag=1")
    a = ap.parse_args()

    if a.split == "test" and not a.final:
        raise SystemExit("El período TEST es el held-out final: agrega --final cuando el agente ya esté definido.")

    cfg, data = load_run(a.run, a.set)
    if a.cost_bps is not None:
        cfg["env"]["fee_bps"], cfg["env"]["slippage_bps"] = a.cost_bps, 0.0
    set_seed(cfg["seed"])
    agent = load_agent(cfg, data, os.path.join(a.run, a.ckpt))
    recs = evaluate_all(cfg, data[a.split], agent, n_random=a.random_seeds)

    suffix = f"_cost{a.cost_bps}" if a.cost_bps is not None else ""
    out = os.path.join(a.run, f"eval_{a.split}{suffix}")
    os.makedirs(out, exist_ok=True)
    df, sd = summarize(recs, cfg["evaluation"]["bars_per_year"])
    df.to_csv(os.path.join(out, "metrics.csv"))
    md = to_markdown(df, sd)
    with open(os.path.join(out, "metrics.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    plot_equity(recs, os.path.join(out, "equity_curves.png"), f"Curva de capital — {a.split}")
    plot_positions(recs["Agente RL"], os.path.join(out, "positions.png"))
    plot_tod_profile(recs["Agente RL"], os.path.join(out, "position_by_time_of_day.png"))
    np.savez_compressed(os.path.join(out, "agent_record.npz"), **{k: v for k, v in recs["Agente RL"].items()})
    print(md)
    print("\nGuardado en", out)


if __name__ == "__main__":
    main()
