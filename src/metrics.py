"""Ejecución de episodios completos y métricas de desempeño."""
from __future__ import annotations

import numpy as np


def run_episode(env, agent, seed: int | None = None) -> dict:
    """Corre un episodio completo y devuelve el registro (curva de capital, posiciones, costos)."""
    obs, info = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    start = env.start
    equity, pos, cost, turn, acts = [env.initial], [0.0], [], [], []
    done = False
    while not done:
        a = agent.act(obs, env)
        obs, _, terminated, truncated, info = env.step(a)
        equity.append(info["equity"])
        pos.append(info["position"])
        cost.append(info["cost"])
        turn.append(info["turnover"])
        acts.append(a)
        done = terminated or truncated
    n = len(equity)
    return {
        "equity": np.asarray(equity), "position": np.asarray(pos), "cost": np.asarray(cost),
        "turnover": np.asarray(turn), "action": np.asarray(acts),
        "ts": env.split.ts[start:start + n], "close": env.close[start:start + n],
    }


def compute_metrics(rec: dict, bars_per_year: float = 98280.0) -> dict:
    eq = rec["equity"]
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    step = np.diff(eq) / eq[:-1]
    sd = step.std()
    sharpe = float(step.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0
    pos = rec["position"][1:]
    return {
        "retorno_acumulado": float(eq[-1] / eq[0] - 1.0),
        "max_drawdown": mdd,
        "costos_total": float(rec["cost"].sum()),
        "costos_pct_capital": float(rec["cost"].sum() / eq[0]),
        "n_cambios_posicion": int((rec["turnover"] > 0).sum()),
        "turnover_total": float(rec["turnover"].sum()),
        "sharpe_anualizado": sharpe,
        "pct_tiempo_invertido": float((pos > 0).mean()),
        "posicion_media": float(pos.mean()),
        "capital_final": float(eq[-1]),
    }


def selection_score(metrics: dict, kind: str) -> float:
    """Métrica usada (en VALIDACIÓN) para elegir el mejor checkpoint."""
    if kind == "calmar":
        return metrics["retorno_acumulado"] / max(abs(metrics["max_drawdown"]), 1e-3)
    return metrics["retorno_acumulado"]
