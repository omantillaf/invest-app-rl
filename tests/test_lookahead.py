"""Auditoría anti-lookahead: los indicadores en t solo dependen de barras <= t."""
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import (apply_scaler, compute_features, feature_names, fit_scaler, make_synthetic,
                      prepare_data, temporal_split)

CFG = load_config("configs/default.yaml")
F = CFG["features"]
COLS = feature_names(F)


def _feats(df):
    return compute_features(df, F, CFG["data"]["session_start"], CFG["data"]["session_minutes"])


def test_features_do_not_use_future_rows():
    df = make_synthetic(n_days=6, seed=1)
    full = _feats(df)
    rng = np.random.default_rng(0)
    for t in rng.integers(0, len(df), size=25):
        trunc = _feats(df.iloc[: t + 1])          # el futuro NO existe
        np.testing.assert_allclose(trunc.loc[t, COLS].to_numpy(float), full.loc[t, COLS].to_numpy(float),
                                   rtol=1e-9, atol=1e-12)


def test_perturbing_future_prices_does_not_change_past_features():
    df = make_synthetic(n_days=6, seed=2)
    base = _feats(df)
    t = 1000
    pert = df.copy()
    pert.loc[t + 1:, ["open", "high", "low", "close", "volume"]] *= 1.7
    after = _feats(pert)
    np.testing.assert_allclose(base.loc[:t, COLS].to_numpy(float), after.loc[:t, COLS].to_numpy(float))


def test_scaler_is_fit_on_train_only():
    df = make_synthetic(n_days=40, seed=3)
    frame = _feats(df)
    parts = temporal_split(frame, CFG["data"]["split"])
    s1 = fit_scaler(parts["train"], COLS)
    # alterar val/test no debe cambiar el scaler
    tampered = {k: v.copy() for k, v in parts.items()}
    tampered["val"][COLS] = tampered["val"][COLS] * 100
    tampered["test"][COLS] = tampered["test"][COLS] * 100
    s2 = fit_scaler(tampered["train"], COLS)
    assert s1 == s2


def test_split_is_chronological_and_disjoint():
    cfg = load_config("configs/default.yaml", ["data.synthetic.n_days=60"])
    d = prepare_data(cfg)
    assert d["train"].ts.max() < d["val"].ts.min() < d["val"].ts.max() < d["test"].ts.min()
    assert not set(d["train"].day_id) & set(d["val"].day_id)
    assert not set(d["val"].day_id) & set(d["test"].day_id)
