import random
import collections
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sliks import DrivingGame

import random
import math
import os
from collections import deque, namedtuple
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# --- Q-Network ---
class QNetwork(nn.Module):
    def __init__(self, input_dim: int , output_dim: int , hidden_sizes: List[int]):
        super(QNetwork, self).__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.model = nn.Sequential(*layers)
    
        for m in self.model:
            if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# --- Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# --- DQN Agent ---
class DQNAgent:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        batch_size: int,
        buffer_capacity: int,
        eps_decay_steps: int,
        device: torch.device = None,
        gamma: float = 0.99,
        lr: float = 1e-4,        
        target_update: int = 1000,
        tau: float = 1.0,
        eps_start: float = 1.0,
        eps_end: float = 0.01        
    ):
        
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.policy_net = QNetwork(input_dim, output_dim, hidden_sizes).to(self.device)
        self.target_net = QNetwork(input_dim, output_dim, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_capacity)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.tau = tau  # if <1, will do soft update
        self.step_count = 0

        # Epsilon-greedy schedule
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = max(1, eps_decay_steps)

        self.output_dim = output_dim

        # Loss
        self.loss_fn = nn.MSELoss()

#----------------------------------------------------------------------------------------------------------------------
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        eps = self.epsilon()
        if eval_mode or random.random() > eps:
            with torch.no_grad():
                qvals = self.policy_net(state_t)
                return int(qvals.argmax(dim=1).item())
        else:
            return random.randrange(self.output_dim)
#----------------------------------------------------------------------------------------------------------------------
    def epsilon(self) -> float:
        frac = min(1.0, self.step_count / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)
    
#----------------------------------------------------------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        self.replay.push(state, action, float(reward), next_state, bool(done))

#----------------------------------------------------------------------------------------------------------------------
    def optimize(self, gradient_clip: float = None):
        """
        Preleva un batch dal replay buffer e applica un aggiornamento dei pesi.
        Usa target network per il target Q-value.
        """
        if len(self.replay) < self.batch_size:
            return None  # niente aggiornamento possibile

        batch = self.replay.sample(self.batch_size)
        # Convert to tensors
        state_batch = torch.tensor(np.vstack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor([b.action for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_batch = torch.tensor(np.vstack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Target Q
        with torch.no_grad():
            next_q_values = self.target_net(next_batch)
            max_next_q, _ = next_q_values.max(dim=1, keepdim=True)
            target_q = reward_batch + (1.0 - done_batch) * (self.gamma * max_next_q)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), gradient_clip)
        self.optimizer.step()

        # Update step count and possibly target network
        self.step_count += 1
        if self.tau >= 1.0:
            # hard update every target_update steps
            if self.step_count % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            # soft update
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()
#----------------------------------------------------------------------------------------------------------------------
    def save(self, file: str):
        """
        Salva stato agent (policy net, optimizer, step_count, eps schedule).
        """
        payload = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'eps_start': self.eps_start,
            'eps_end': self.eps_end,
            'eps_decay_steps': self.eps_decay_steps,
        }
        os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
        torch.save(payload, file)

#----------------------------------------------------------------------------------------------------------------------
    def load(self, file: str):
        """
        Carica stato agent (se presente).
        """
        try:
            data = torch.load(file, map_location=self.device)
            self.policy_net.load_state_dict(data['policy_state_dict'])
            self.target_net.load_state_dict(data.get('target_state_dict', data['policy_state_dict']))
            if 'optimizer_state_dict' in data:
                try:
                    self.optimizer.load_state_dict(data['optimizer_state_dict'])
                except Exception:
                    # optimizer state might be incompatible across devices/versions; ignore if fails
                    pass
            self.step_count = data.get('step_count', 0)
            self.eps_start = data.get('eps_start', self.eps_start)
            self.eps_end = data.get('eps_end', self.eps_end)
            self.eps_decay_steps = data.get('eps_decay_steps', self.eps_decay_steps)
            self.eps_start = max(self.eps_start, 0.2)
            print(f"Agent load {file}");
        except Exception:
            pass

#----------------------------------------------------------------------------------------------------------------------
# --- Training loop ---
def train_dqn(
    agent: DQNAgent,
    game,
    num_episodes: int,
    max_steps_per_episode: int,
    reward_scale: float = 1.0,
    model_path: str = None):
    """
    Train loop generico. reward_scale consente di normalizzare ricompense eventualmente grandi.
    Assume reward in [0, 10000] come indicato; si può ridurre con reward_scale=10000.0 per stabilità.
    """
    agent.load(model_path)
    
    for ep in range(1, num_episodes + 1):
        game.reset()
        state = game.getState()
        ep_reward = 0.0
        for t in range(max_steps_per_episode):
            action = agent.select_action(state, eval_mode=False)
            game.setAction(action)
            next_state, reward, done, info = game.step()
            # normalize reward optionally
            r = float(reward) / reward_scale
            agent.store_transition(state, action, r, next_state, done)
            loss = agent.optimize()
            state = next_state
            ep_reward += float(reward)
            if done:
                break
        print(f"Episode {ep}/{num_episodes} t: {t} avg_reward: {ep_reward:.2f}  eps: {agent.epsilon():.3f}  buffer: {agent.replay.count()}")

        if model_path and ep % 25 == 0:
            print(f"Agent save {model_path}");
            agent.save(model_path)


#----------------------------------------------------------------------------------------------------------------------
# --- Testing loop ---
def test_dqn(agent: DQNAgent, game, 
             num_episodes: int = 10,
             max_steps_per_episode: int = 1000, 
             model_path: str = None):
    """
    Valuta l'agente in modalità greedy (no epsilon exploration).
    Restituisce lista delle ricompense total per episodio.
    """
    agent.load(model_path)

    results = []
    for ep in range(num_episodes):
        state = game.reset()
        state = game.getState()
        ep_reward = 0.0
        for t in range(max_steps_per_episode):
            action = agent.select_action(state, eval_mode=True)
            game.setAction(action)
            next_state, reward, done, info = game.step()
            ep_reward += float(reward)
            state = next_state
            if done:
                break
        results.append(ep_reward)
    return results
#----------------------------------------------------------------------------------------------------------------------


