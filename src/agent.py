import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
from src.config import Config

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64]):
        super().__init__()
        layers = []
        input_dim = state_dim

        for h_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, hparams, device=None, deque_size=5):
        self.device = device or Config.DEVICE
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Networks
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.requires_grad_(False)

        # Experience replay
        self.replay_buffer = ReplayBuffer(Config.REPLAY_CAPACITY)

        # Fitness tracking (Moving average of last 'deque_size' episodes)
        self.recent_rewards = deque(maxlen=deque_size)

        # Hyperparameters
        self.hparams = hparams.copy()
        self.total_steps = 0
        self._apply_hparams()

    def _apply_hparams(self):
        """Apply hyperparameters to optimizer"""
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.hparams["lr"])
        self.batch_size = int(self.hparams["batch_size"])
        self.eps_decay = self.hparams["epsilon_decay"]

    def get_hparams(self):
        return self.hparams.copy()

    def set_hparams(self, new_hparams):
        self.hparams = new_hparams.copy()
        self._apply_hparams()

    def load_weights(self, state_dict):
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def act(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy"""
        if eval_mode:
            eps = 0.0
        else:
            eps = 0.01 + 0.99 * math.exp(-self.total_steps / self.eps_decay)

        if random.random() < eps:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_net(s).argmax().item()

    def train_step(self):
        """Single training step"""
        if len(self.replay_buffer) < self.batch_size:
            return

        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)

        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(-1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)

        q = self.q_net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_net(ns).max(1, keepdim=True)[0]
            target = r + Config.GAMMA * (1 - d) * q_next

        loss = nn.MSELoss()(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % Config.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())