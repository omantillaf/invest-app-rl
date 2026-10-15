"""Entrenamiento DQN. La validación (NO el test) decide el mejor checkpoint y la parada anticipada.

    python -m src.train --config configs/default.yaml --reward v1_return --seed 0 --out runs/v1_return_s0
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from .agent import DQNAgent
from .config import clone, eval_cfg, load_config, save_config, set_seed
from .data import prepare_data, save_scaler
from .environment import TradingEnv
from .metrics import compute_metrics, run_episode, selection_score


def train(cfg: dict, run_dir: str, verbose: bool = True, data: dict | None = None) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    seed = cfg["seed"]
    set_seed(seed)
    c = cfg["dqn"]
    data = data or prepare_data(cfg)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    save_scaler(data["scaler"], os.path.join(run_dir, "scaler.json"))

    env = TradingEnv(data["train"], cfg, episode_len=c["episode_len"], random_start=True)
    val_env = TradingEnv(data["val"], eval_cfg(cfg))
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg, seed)
    bpy = cfg["evaluation"]["bars_per_year"]

    obs, _ = env.reset(seed=seed)
    decay_steps = max(1, int(c["total_steps"] * c["eps_decay_fraction"]))
    best, best_step, bad, log = -np.inf, 0, 0, []
    ep_ret, ep_rets, losses = 0.0, [], []
    t0 = time.time()

    for step in range(1, c["total_steps"] + 1):
        eps = max(c["eps_end"], c["eps_start"] - (c["eps_start"] - c["eps_end"]) * step / decay_steps)
        a = agent.select_action(obs, eps)
        obs2, r, term, trunc, _ = env.step(a)
        agent.buffer.add(obs, a, r, obs2, float(term))   # solo 'terminated' corta el bootstrap
        obs = obs2
        ep_ret += r
        if term or trunc:
            ep_rets.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
        if step >= c["learn_start"] and step % c["train_freq"] == 0 and len(agent.buffer) >= c["batch_size"]:
            losses.append(agent.learn())
        if step % c["target_update"] == 0:
            agent.sync()

        if step % c["eval_every"] == 0 or step == c["total_steps"]:
            m = compute_metrics(run_episode(val_env, agent), bpy)
            score = selection_score(m, c["selection_metric"])
            row = {"step": step, "epsilon": round(eps, 4), "loss": float(np.mean(losses[-200:])) if losses else np.nan,
                   "train_ep_reward": float(np.mean(ep_rets[-20:])) if ep_rets else np.nan,
                   "val_return": m["retorno_acumulado"], "val_max_dd": m["max_drawdown"],
                   "val_trades": m["n_cambios_posicion"], "val_pct_invested": m["pct_tiempo_invertido"],
                   "val_score": score}
            log.append(row)
            if verbose:
                print(f"[{step:>7}] eps={eps:.2f} loss={row['loss']:.4f} val_ret={m['retorno_acumulado']:+.2%} "
                      f"dd={m['max_drawdown']:.2%} trades={m['n_cambios_posicion']} "
                      f"inv={m['pct_tiempo_invertido']:.0%}  ({time.time() - t0:.0f}s)", flush=True)
            # Solo se elige checkpoint / cuenta la paciencia cuando ya terminó el decaimiento de ε.
            # Antes, un checkpoint temprano casi sin entrenar que se queda 100 % fuera del mercado
            # (retorno 0 %) ganaba por defecto contra políticas que operaban y perdían un poco.
            if step >= decay_steps:
                if score > best:
                    best, best_step, bad = score, step, 0
                    agent.save(os.path.join(run_dir, "best.pt"))
                else:
                    bad += 1
                if bad >= c["patience"]:
                    if verbose:
                        print(f"Parada anticipada: sin mejora en validación durante {bad} evaluaciones.")
                    break

    agent.save(os.path.join(run_dir, "final.pt"))
    if not os.path.exists(os.path.join(run_dir, "best.pt")):
        agent.save(os.path.join(run_dir, "best.pt"))
    pd.DataFrame(log).to_csv(os.path.join(run_dir, "log.csv"), index=False)
    return {"best_val_score": best, "best_step": best_step, "steps_run": step}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--reward", default=None, help="v1_return | v2_turnover | v3_risk")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[], help="override, p. ej. dqn.lr=0.0003")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.reward:
        cfg["reward"]["name"] = a.reward
    if a.seed is not None:
        cfg["seed"] = a.seed
    out = a.out or f"runs/{cfg['reward']['name']}_s{cfg['seed']}"
    print(train(cfg, out))


if __name__ == "__main__":
    main()