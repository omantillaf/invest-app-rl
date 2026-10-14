"""Carga de configuración YAML, overrides por CLI y fijación de semillas."""
from __future__ import annotations

import copy
import os
import random

import numpy as np
import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def clone(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def eval_cfg(cfg: dict) -> dict:
    """Config para EVALUAR: sin terminación por ruina, así todas las estrategias recorren
    el período completo y son comparables (la regla de ruina solo aplica al entrenamiento)."""
    c = copy.deepcopy(cfg)
    c["env"]["ruin_fraction"] = 0.0
    return c


def set_seed(seed: int) -> None:
    """Fija semillas de Python, NumPy y PyTorch (si está instalado)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def save_config(cfg: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
