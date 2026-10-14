"""Agente DQN (con opción Double DQN), replay buffer y red MLP."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int):
        self.cap, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done):
        self.s[self.i], self.a[self.i], self.r[self.i] = s, a, r
        self.s2[self.i], self.d[self.i] = s2, done
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.n


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, seed: int = 0):
        c = cfg["dqn"]
        self.cfg, self.n_actions, self.obs_dim = c, n_actions, obs_dim
        dev = c.get("device", "cpu")
        self.device = torch.device("cuda" if dev == "auto" and torch.cuda.is_available() else
                                   ("cpu" if dev == "auto" else dev))
        self.q = QNetwork(obs_dim, n_actions, c["hidden"]).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, c["hidden"]).to(self.device)
        self.sync()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=c["lr"])
        self.buffer = ReplayBuffer(c["buffer_size"], obs_dim, seed)
        self.rng = np.random.default_rng(seed + 1)

    def sync(self):
        self.target.load_state_dict(self.q.state_dict())

    @torch.no_grad()
    def act(self, obs, env=None) -> int:
        """Política greedy (evaluación). `env` se ignora; existe por compatibilidad con baselines."""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q(x).argmax(dim=1).item())

    def select_action(self, obs, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return self.act(obs)

    def learn(self) -> float:
        c = self.cfg
        s, a, r, s2, d = (torch.as_tensor(x, device=self.device) for x in self.buffer.sample(c["batch_size"]))
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if c["double"]:
                a2 = self.q(s2).argmax(dim=1, keepdim=True)
                q2 = self.target(s2).gather(1, a2).squeeze(1)
            else:
                q2 = self.target(s2).max(dim=1).values
            y = r + c["gamma"] * (1.0 - d) * q2
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c["grad_clip"])
        self.opt.step()
        return float(loss.item())

    def save(self, path: str):
        torch.save({"q": self.q.state_dict(), "obs_dim": int(self.obs_dim), "n_actions": int(self.n_actions)}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ck["q"])
        self.sync()
        return self
