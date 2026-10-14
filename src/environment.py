"""Entorno Gymnasium de inversión intradía (scalping), lookahead-safe.

Convención de tiempo (importante para el informe)
-------------------------------------------------
En el índice t el agente ve:   features[t]  (calculados SOLO con barras <= t)  + posición actual.
Decide la posición objetivo p' en el CIERRE de la barra t y la ejecuta a close[t].
La ganancia de ese paso es el movimiento  close[t] -> close[t+1]:

    V_net = V_t * (1 - c * |p' - p|)               # costo de transacción sobre el turnover
    V_t+1 = V_net * (1 + p' * (close[t+1]/close[t] - 1))
    reward = escala * f(V_t, V_t+1, ...)            # ver _reward()

Con execution_lag = 1 la orden decidida en t se ejecuta en close[t+1] (más realista; se
recomienda como prueba de robustez): el paso t usa la posición decidida en t-1.

Scalping: si flat_at_close = True, en la última barra de cada día la posición se fuerza a 0
(se paga el costo de cerrar) y no se gana el gap overnight.
Se asume rebalanceo a la fracción objetivo en cada barra; el costo se cobra sobre |p' - p|.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import SplitData


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, split: SplitData, cfg: dict, episode_len: int | None = None,
                 random_start: bool = False, flat_at_close: bool | None = None):
        super().__init__()
        ecfg, rcfg = cfg["env"], cfg["reward"]
        self.split = split
        self.features = split.features
        self.close = split.close
        self.last_of_day = split.last_of_day
        self.T = len(self.close)
        self.levels = np.asarray(ecfg["action_levels"], dtype=np.float64)
        self.cost_rate = (ecfg["fee_bps"] + ecfg["slippage_bps"]) / 1e4
        self.initial = float(ecfg["initial_capital"])
        self.flat_at_close = ecfg["flat_at_close"] if flat_at_close is None else flat_at_close
        self.lag = int(ecfg.get("execution_lag", 0))
        self.ruin = float(ecfg["ruin_fraction"]) * self.initial
        self.rname = rcfg["name"]
        self.rscale = float(rcfg["scale"])
        self.lam_turn = float(rcfg.get("turnover_penalty", 0.0))
        self.lam_dd = float(rcfg.get("drawdown_penalty", 0.0))
        self.episode_len = episode_len
        self.random_start = random_start
        self.action_space = spaces.Discrete(len(self.levels))
        n_obs = self.features.shape[1] + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        self.t = 0
        self.start = 0
        self.end = self.T - 1

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        pos = self.pending if self.lag else self.pos
        return np.concatenate([self.features[self.t], [pos]]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.random_start and self.episode_len:
            hi = max(self.T - self.episode_len - 1, 1)
            self.start = int(self.np_random.integers(0, hi))
        else:
            self.start = 0
        self.t = self.start
        self.end = min(self.T - 1, self.start + self.episode_len) if self.episode_len else self.T - 1
        self.equity = self.initial
        self.peak = self.initial
        self.dd = 0.0
        self.pos = 0.0       # posición efectivamente mantenida
        self.pending = 0.0   # última posición objetivo (solo distinta de pos si lag = 1)
        return self._obs(), {"equity": self.equity, "position": 0.0, "t": self.t}

    def _reward(self, v0: float, v1: float, turn: float, dd_new: float) -> float:
        if self.rname == "v1_return":
            r = (v1 - v0) / v0
        elif self.rname == "v2_turnover":
            r = np.log(v1 / v0) - self.lam_turn * turn
        elif self.rname == "v3_risk":
            r = np.log(v1 / v0) - self.lam_turn * turn - self.lam_dd * max(0.0, dd_new - self.dd)
        else:
            raise ValueError(f"Recompensa desconocida: {self.rname}")
        return float(self.rscale * r)

    def step(self, action: int):
        t = self.t
        target = float(self.levels[int(action)])
        forced = bool(self.flat_at_close and self.last_of_day[t])
        if forced:
            effective, self.pending = 0.0, 0.0
        elif self.lag:
            effective, self.pending = self.pending, target
        else:
            effective = target
            self.pending = target

        turn = abs(effective - self.pos)
        v0 = self.equity
        cost = v0 * self.cost_rate * turn
        r_bar = self.close[t + 1] / self.close[t] - 1.0
        v1 = (v0 - cost) * (1.0 + effective * r_bar)

        self.peak = max(self.peak, v1)
        dd_new = 1.0 - v1 / self.peak
        reward = self._reward(v0, v1, turn, dd_new)
        self.dd = dd_new
        self.equity, self.pos = v1, effective
        self.t += 1

        terminated = bool(v1 < self.ruin)
        truncated = bool(self.t >= self.end)
        info = {"equity": v1, "position": effective, "cost": cost, "turnover": turn,
                "forced_flat": forced, "t": self.t, "bar_return": r_bar}
        return self._obs(), reward, terminated, truncated, info
