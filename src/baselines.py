"""Baselines: aleatorio, comprar y mantener (dos variantes) y cruce de medias móviles.

Todos exponen  .act(obs, env) -> índice de acción  y se evalúan en el MISMO entorno
(mismos datos, mismos costos) que el agente RL.
"""
from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, n_actions: int, seed: int = 0):
        self.n, self.rng = n_actions, np.random.default_rng(seed)

    def act(self, obs, env=None) -> int:
        return int(self.rng.integers(0, self.n))


class BuyAndHold:
    """Siempre 100 % invertido. Se evalúa de dos formas (ver evaluate.py):
    * 'Comprar y mantener' (definición del enunciado): entorno con flat_at_close=False
      -> una sola compra al inicio, incluye gaps overnight.
    * 'B&H intradía': mismo entorno del agente (cierra cada día) -> comparación homogénea.
    """

    def __init__(self, n_actions: int):
        self.a = n_actions - 1  # nivel más alto de exposición

    def act(self, obs, env=None) -> int:
        return self.a


class MovingAverageCross:
    """Baseline clásico: 100 % invertido si SMA rápida > SMA lenta, si no 0 %. Usa solo precios <= t."""

    def __init__(self, n_actions: int, fast: int = 10, slow: int = 30):
        self.hi, self.fast, self.slow = n_actions - 1, fast, slow

    def act(self, obs, env=None) -> int:
        t = env.t
        if t + 1 < self.slow:
            return 0
        c = env.close
        return self.hi if c[t - self.fast + 1:t + 1].mean() > c[t - self.slow + 1:t + 1].mean() else 0
