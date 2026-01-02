import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tris import Tris, Action

Transition = collections.namedtuple('Transition', ('state','action','reward','next_state','done'))

# stato: vettore 9 float (+1 agent, -1 opp, 0 empty)
def board_to_tensor(board, agent_mark=1):
    arr = np.zeros(9, dtype=np.float32)
    b = np.array(board).reshape(9)
    arr[b == agent_mark] = 1.0
    arr[(b != 0) & (b != agent_mark)] = -1.0
    state = torch.from_numpy(arr)
    return state

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
    def __init__(self, device='cpu', explorationRate = 0, lr=1e-3, gamma=0.99, batch_size=64, buffer_capacity=5000):
        self.device = torch.device(device)
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(buffer_capacity)
        self.explorationRate = explorationRate                       # exploration rate, che controlla la probabilità di scegliere un'azione casuale invece dell'azione ottimale.
        self.explorationRate_min = 0.05
        self.explorationRate_decay = 0.9999
        self.update_target_steps = 500
        self.step_count = 0
        self.loss_fn = nn.MSELoss()
#--------------------------------------------------------------------------------------------------------------------
    def select_action(self, state_tensor, available_actions) -> Action:
        if random.random() < self.explorationRate:
            return random.choice(available_actions)
        with torch.no_grad():
            qvals = self.policy_net(state_tensor.to(self.device).unsqueeze(0)).cpu().numpy().ravel()
        mask = np.full(9, -np.inf)
        move = {}
        for action in available_actions:
            idx = action.getIdx( 3 )
            move[idx] = action
            mask[idx] = qvals[idx]
        best_idx = int(np.argmax(mask))
        return move[best_idx]
#--------------------------------------------------------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        aidx = action.getIdx( 3 )
        self.replay.push(state.numpy(), aidx, reward, None if next_state is None else next_state.numpy(), done)
#--------------------------------------------------------------------------------------------------------------------
    def optimize(self):
        if len(self.replay) < self.batch_size:
            return
        
        # Sample batch e prepara tensori
        batch = self.replay.sample(self.batch_size)

        states = torch.tensor(np.stack(batch.state)).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Calcola Q-value attuale
        q_values = self.policy_net(states).gather(1, actions)

        # Calcola target Q-value
        with torch.no_grad():
            # Filtra solo stati non-terminali per calcolo next_q
            non_final_mask = torch.tensor([ns is not None for ns in batch.next_state], dtype=torch.bool).to(self.device)
            non_final_next_states = torch.tensor(
                np.stack([ns for ns in batch.next_state if ns is not None])
            ).to(self.device)
            
            # Calcola max Q per stati non-terminali
            next_q_all = torch.zeros(len(batch.state), 1, device=self.device)
            if non_final_next_states.numel() > 0:
                next_q_vals = self.target_net(non_final_next_states).max(1)[0]
                next_q_all[non_final_mask] = next_q_vals.unsqueeze(1)
            
            target = rewards + (1.0 - dones) * self.gamma * next_q_all

        # Ottimizzazione
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network e decay epsilon
        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())       # Aggiorna la rete target con i pesi della rete policy

        if self.explorationRate > self.explorationRate_min:
            self.explorationRate *= self.explorationRate_decay
#--------------------------------------------------------------------------------------------------------------------
    def save(self, path):
        torch.save({'policy': self.policy_net.state_dict(),
                    'target': self.target_net.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)
#--------------------------------------------------------------------------------------------------------------------
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.target_net.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])


###############################################################################################################################################
def _play_agent_turn(game, agent, state):

    """Esegui turno agente. Ritorna (action, reward, done, next_state)"""
    avail_actions = game.available_actions()
    action = agent.select_action(state, avail_actions)
    
    if not game.make_move_action(action):
        print(f"Action not valid: {action}")
        action = random.choice(avail_actions)
        game.make_move_action(action)
    
    if game.winner == 'X':                  
        return action, 1.0, True, None
    elif not game.available_moves():
        return action, 0.1, True, None
    else:
        return action, 0.0, False, board_to_tensor(game.board, agent_mark=1)

def _play_opponent_turn(game, agent, state):
    """Esegui turno avversario. Ritorna (reward, done, next_state)"""
    opp_action = random.choice(game.available_actions())
    game.make_move_action(opp_action)
    
    if game.winner == 'O':
        return -1.0, True, None
    elif not game.available_moves():
        return 0.1, True, None
    else:
        return 0.0, False, board_to_tensor(game.board, agent_mark=1)

###############################################################################################################################################
# Training loop: integra con la classe Tris presente in tris.py
def train_dqn(agent, num_episodes, opponent='random'):

    for ep in range(1, num_episodes + 1):
        game = Tris()
        game.reset()
        state = board_to_tensor(game.board, agent_mark=1)

        while (not game.game_over) and game.available_actions():
            if game.current_player == game.players[0]:           # Turno agente        
                action, reward, done, next_state = _play_agent_turn(game, agent, state)
                agent.remember(state, action, reward, next_state, done)
                agent.optimize()
                if done:
                    break
                state = next_state
            else:                                               # Turno avversario                
                reward, done, next_state = _play_opponent_turn(game, agent, state)
                agent.optimize()
                if done:
                    break
                state = next_state

        if ep % 500 == 0:
            print(f"Episode {ep}, explorationRate {agent.explorationRate:.3f}, buffer {len(agent.replay)}")
            #agent.save(f"dqn_tris_ep{ep}.pth")

            wins, draws, losses = test_dqn(agent, 100)
            # Calcola e stampa i risultati
            total = wins + draws + losses
            accuracy = (wins / total) * 100 if total > 0 else 0
            print(f"  Vittorie:  {wins:3d} ({wins/total*100:5.1f}%)")
            print(f"  Pareggi:   {draws:3d} ({draws/total*100:5.1f}%)")
            print(f"  Sconfitte: {losses:3d} ({losses/total*100:5.1f}%)")
            print(f"  ACCURATEZZA: {accuracy:.1f}%")

###############################################################################################################################################
def test_dqn(agent, num_games=100):
    """
    Test l'agente DQN contro un avversario random per num_games partite.
    Ritorna il numero di vittorie, pareggi e sconfitte.
    """
    wins = 0
    draws = 0
    losses = 0

    for game_idx in range(num_games):
        game = Tris()
        game.reset()
        state = board_to_tensor(game.board, agent_mark=1)

        while (not game.game_over) and (bool(game.available_actions())):
            if agent and game.current_player == game.players[0]:           # Agent (X)
                avail = game.available_actions()
                action = agent.select_action(state, avail)
                if not game.make_move_action(action):
                    print(f"Action not valid: {action}")  
                    action = random.choice(avail)
                    game.make_move_action(action)
            else:  # Opponent (O) - random
                opp_action = random.choice(game.available_actions())       # Player (O)
                game.make_move_action(opp_action)

            # Update state after opponent's move
            if not game.game_over and bool(game.available_actions()):
                state = board_to_tensor(game.board, agent_mark=1)

        # Count result
        if game.winner == 'X':
            wins += 1
        elif game.winner == 'O':
            losses += 1
        else:
            draws += 1

    print(f"Partite completate: {game_idx + 1}/{num_games}")
    return wins, draws, losses
