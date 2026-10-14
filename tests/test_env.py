"""Pruebas de la mecánica del entorno: costos, timing, cierre diario y recompensa."""
import numpy as np

from src.config import load_config, clone
from src.data import SplitData
from src.environment import TradingEnv


def _split(close, per_day):
    close = np.asarray(close, float)
    n = len(close)
    day = np.arange(n) // per_day
    last = np.r_[day[1:] != day[:-1], True]
    return SplitData("t", np.zeros((n, 2), np.float32), close, np.arange(n).astype("datetime64[m]"), day, last)


def _cfg(**env):
    cfg = clone(load_config("configs/default.yaml"))
    cfg["env"].update({"fee_bps": 5.0, "slippage_bps": 5.0, "flat_at_close": False})
    cfg["env"].update(env)
    cfg["reward"].update({"name": "v1_return", "scale": 1.0})
    return cfg


def _run(env, actions):
    env.reset()
    for a in actions:
        _, r, te, tr, info = env.step(a)
        if te or tr:
            break
    return env


def test_always_flat_keeps_capital_and_pays_no_cost():
    env = _run(TradingEnv(_split(np.linspace(100, 110, 50), 50), _cfg()), [0] * 49)
    assert env.equity == env.initial


def test_buy_and_hold_matches_closed_form():
    close = np.linspace(100, 110, 50)
    cfg = _cfg()
    env = _run(TradingEnv(_split(close, 50), cfg), [2] * 49)
    c = (cfg["env"]["fee_bps"] + cfg["env"]["slippage_bps"]) / 1e4
    expected = env.initial * (1 - c) * close[-1] / close[0]
    assert abs(env.equity - expected) < 1e-6


def test_decision_uses_next_bar_return_only():
    # el retorno de la barra t->t+1 es el único que gana/pierde la posición decidida en t
    close = [100, 110, 110, 110]
    cfg = _cfg(fee_bps=0.0, slippage_bps=0.0)
    env = TradingEnv(_split(close, 4), cfg)
    env.reset()
    _, _, _, _, info = env.step(2)           # decide en t=0 -> gana 100->110
    assert abs(info["equity"] - env.initial * 1.10) < 1e-6
    _, _, _, _, info = env.step(2)           # 110->110: sin cambio
    assert abs(info["equity"] - env.initial * 1.10) < 1e-6


def test_transaction_costs_on_turnover():
    cfg = _cfg(fee_bps=10.0, slippage_bps=0.0)
    env = TradingEnv(_split(np.full(20, 100.0), 20), cfg)
    env.reset()
    total = 0.0
    for a in [2, 0, 2, 1, 1]:                # turnover: 1, 1, 1, .5, 0
        _, _, _, _, info = env.step(a)
        total += info["cost"]
    assert info["turnover"] == 0.0
    expected = env.initial * 0.001 * (1 + 1 + 1 + 0.5)  # aproximado: el capital baja levemente con cada costo
    assert 0.99 * expected < total < 1.001 * expected


def test_flat_at_close_forces_zero_and_skips_overnight_gap():
    close = [100, 100, 120, 120]            # gap de 100 -> 120 entre día 1 y día 2
    cfg = _cfg(fee_bps=0.0, slippage_bps=0.0, flat_at_close=True)
    env = TradingEnv(_split(close, 2), cfg)
    env.reset()
    env.step(2)                              # t=0 (no es última barra): invertido, ret 0
    _, _, _, _, info = env.step(2)           # t=1 es última del día 1 -> forzado a 0
    assert info["forced_flat"] and info["position"] == 0.0
    assert abs(info["equity"] - env.initial) < 1e-9   # no ganó el gap


def test_execution_lag_delays_position_one_bar():
    close = [100, 110, 121, 121]
    cfg = _cfg(fee_bps=0.0, slippage_bps=0.0, execution_lag=1)
    env = TradingEnv(_split(close, 4), cfg)
    env.reset()
    _, _, _, _, i1 = env.step(2)             # decide invertir, aún no efectiva
    assert i1["equity"] == env.initial
    _, _, _, _, i2 = env.step(2)             # ahora sí gana 110 -> 121
    assert abs(i2["equity"] - env.initial * 1.10) < 1e-6


def test_v1_reward_sums_to_total_return_without_scaling():
    close = np.array([100, 101, 99, 103, 104, 102, 105.0])
    cfg = _cfg(fee_bps=0.0, slippage_bps=0.0)
    env = TradingEnv(_split(close, 7), cfg)
    env.reset()
    prod, done = 1.0, False
    while not done:
        _, r, te, tr, _ = env.step(2)
        prod *= 1 + r
        done = te or tr
    assert abs(prod - close[-1] / close[0]) < 1e-9


def test_eval_config_runs_full_period_even_when_capital_collapses():
    from src.config import eval_cfg
    from src.baselines import RandomAgent
    from src.metrics import run_episode
    cfg = _cfg(fee_bps=200.0, slippage_bps=0.0)       # costo absurdo: la ruina ocurriría en entrenamiento
    split = _split(np.full(400, 100.0), 400)
    short = run_episode(TradingEnv(split, cfg), RandomAgent(3, 0))
    full = run_episode(TradingEnv(split, eval_cfg(cfg)), RandomAgent(3, 0))
    assert len(short["equity"]) < len(full["equity"]) == 400
