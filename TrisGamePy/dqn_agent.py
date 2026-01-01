import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

Transition = collections.namedtuple('Transition', ('state','action','reward','next_state','done'))

# stato: vettore 9 float (+1 agent, -1 opp, 0 empty)
def board_to_tensor(board, agent_mark=1):
    arr = np.zeros(9, dtype=np.float32)
    b = np.array(board).reshape(9)
    arr[b == agent_mark] = 1.0
    arr[(b != 0) & (b != agent_mark)] = -1.0
    return torch.from_numpy(arr)
###############################################################################################################################################
class QNetwork(nn.Module):
    def __init__(self, in_dim=9, out_dim=9, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)
###############################################################################################################################################
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)
###############################################################################################################################################
class DQNAgent:
    def __init__(self, device='cpu', lr=1e-3, gamma=0.99, batch_size=64, buffer_capacity=5000):
        self.device = torch.device(device)
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(buffer_capacity)
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.9995
        self.update_target_steps = 500
        self.step_count = 0
        self.loss_fn = nn.MSELoss()

    def select_action(self, state_tensor, available_moves):
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        with torch.no_grad():
            qvals = self.policy_net(state_tensor.to(self.device).unsqueeze(0)).cpu().numpy().ravel()
        mask = np.full(9, -np.inf)
        for (i,j) in available_moves:
            idx = i*3 + j
            mask[idx] = qvals[idx]
        best_idx = int(np.argmax(mask))
        return (best_idx // 3, best_idx % 3)

    def remember(self, state, action, reward, next_state, done):
        aidx = action[0]*3 + action[1]
        self.replay.push(state.numpy(), aidx, reward, None if next_state is None else next_state.numpy(), done)

    def optimize(self):
        if len(self.replay) < self.batch_size:
            return
        batch = self.replay.sample(self.batch_size)
        states = torch.tensor(np.stack(batch.state)).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = None
        if batch.next_state[0] is not None:
            next_states = torch.tensor(np.stack(batch.next_state)).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            if next_states is not None:
                next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            else:
                next_q = torch.zeros_like(q_values)
            target = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

    def save(self, path):
        torch.save({'policy': self.policy_net.state_dict(),
                    'target': self.target_net.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.target_net.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

# Training loop: integra con la classe Tris presente in TrisGame/tris.py
def train_dqn(agent, num_episodes, opponent='random'):
    from TrisGame.tris import Tris
    for ep in range(1, num_episodes+1):
        game = Tris()
        game.reset()
        game.current_player = game.players[0]
        state = board_to_tensor(game.board, agent_mark=1)
        done = False

        while (not game.game_over) and (bool(game.available_moves())):
            if game.current_player == game.players[0]:
                avail = game.available_moves()
                action = agent.select_action(state, avail)
                if not game.make_move(action):
                    action = random.choice(avail)
                    game.make_move(action)

                if game.winner == 'X':
                    reward = 1.0
                    done = True
                    next_state = None
                elif game.winner == 'O':
                    reward = -1.0
                    done = True
                    next_state = None
                elif not game.available_moves():
                    reward = 0.0
                    done = True
                    next_state = None
                else:
                    reward = 0.0
                    next_state = board_to_tensor(game.board, agent_mark=1)

                agent.remember(state, action, reward, None if next_state is None else next_state, done)
                state = next_state if next_state is not None else state
                agent.optimize()
                if done:
                    break

            else:
                if opponent == 'random':
                    opp_action = random.choice(game.available_moves())
                else:
                    opp_action = random.choice(game.available_moves())
                game.make_move(opp_action)
                if game.winner == 'O':
                    agent.remember(state, action, -1.0, None, True)
                    agent.optimize()
                    break
                elif not game.available_moves():
                    agent.remember(state, action, 0.0, None, True)
                    agent.optimize()
                    break
                else:
                    state = board_to_tensor(game.board, agent_mark=1)

        if ep % 500 == 0:
            print(f"Episode {ep}, epsilon {agent.epsilon:.3f}, buffer {len(agent.replay)}")
            agent.save(f"dqn_tris_ep{ep}.pth")

###############################################################################################################################################
if __name__ == '__main__':
    a = DQNAgent()
    train_dqn(a, num_episodes=2000)
    a.save('dqn_tris_final.pth')
