import random
import collections
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tris import Tris, Action

Transition = collections.namedtuple('Transition', ('state','action','reward','next_state','done'))

# stato: vettore 9 float (+1 agent, -1 opp, 0 empty)
def board_to_tensor(game, agent_mark=1):
    elem = game.nrows*game.ncols
    arr = np.zeros(elem, dtype=np.float32)
    b = np.array(game.board).reshape(elem)
    arr[b == agent_mark] = 1.0
    arr[(b != 0) & (b != agent_mark)] = -1.0
    state = torch.from_numpy(arr)
    return state

###############################################################################################################################################
class QNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        # Inizializzazione consigliata dei pesi (opzionale ma utile)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

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
    def __init__(self, device, game, explorationRate, lr=1e-5, gamma=0.99, batch_size=128, buffer_capacity=10000):
        self.device = torch.device(device)
        self.game = game
        self.policy_net = QNetwork(in_dim=game.dim() , out_dim=game.dim() ).to(self.device)
        self.target_net = QNetwork(in_dim=game.dim() , out_dim=game.dim() ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(buffer_capacity)
        self.replay_temp = ReplayBuffer(100)
        #EXPLORATION
        self.explorationRate = explorationRate                       # exploration rate, che controlla la probabilità di scegliere un'azione casuale invece dell'azione ottimale.
        self.explorationRate_min = 0.05
        self.explorationRate_decay = 0.9997
        #TARGET NET UPDATE
        self.update_target_steps = 500
        self.step_count = 0
        self.loss_fn = nn.MSELoss()

#--------------------------------------------------------------------------------------------------------------------
    def select_action(self, state_tensor, available_actions) -> Action:
        if random.random() < self.explorationRate:
            return self.game.random_action()
        with torch.no_grad():
            qvals = self.policy_net(state_tensor.to(self.device).unsqueeze(0)).cpu().numpy().ravel()
        max_qv = -1000
        opt_action = None
        for action in available_actions:
            idx = self.game.getIdx( action )
            qv = qvals[idx]
            if qv > max_qv:
                max_qv = qv
                opt_action = action
        return opt_action 
    
#--------------------------------------------------------------------------------------------------------------------
    def print_action_rprobs(self, state_tensor, available_actions):
        with torch.no_grad():
            qvals = self.policy_net(state_tensor.to(self.device).unsqueeze(0)).cpu().numpy().ravel()
        msg = "Q-val: " + " ".join(f"{action.str()}{qvals[self.game.getIdx(action) ]:.3f}" for action in available_actions)
        print(msg[:200].ljust(200), end='\r')

#--------------------------------------------------------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, run,  done):
        aidx =  self.game.getIdx(action)

        if (done == False):
            self.replay_temp.push(state.numpy(), aidx, run, None if next_state is None else next_state.numpy(), False)
        else:
            self.replay.push(state.numpy(), aidx, reward, None if next_state is None else next_state.numpy(), True)
            #Aggiungo tutti gli altri da replay_temp
            for ep in self.replay_temp.buffer:
                self.replay.push(ep.state, ep.action, reward, ep.next_state, True)
            self.replay_temp.buffer.clear()

        
#--------------------------------------------------------------------------------------------------------------------
    def optimize(self):
        # Se non ci sono abbastanza transizioni nel replay buffer, esci subito
        if len(self.replay) < self.batch_size:
            return

        # Preleva un batch dal replay buffer
        batch = self.replay.sample(self.batch_size)

        # Costruisci tensori degli stati, assicurandoti tipo float32 e device corretto
        # np.stack è usato se batch.state contiene array NumPy; altrimenti usare torch.stack
        states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)

        # Azioni come long (indice per gather) e con shape (B,1)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)

        # Ricompense come float con shape (B,1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Done/terminal come float (1.0 se terminale) con shape (B,1)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Calcola i Q attuali dalla policy network e seleziona i valori degli actions presi
        q_values = self.policy_net(states).gather(1, actions)  # shape (B,1)

        # Calcolo target Q-value senza tracciare il gradiente
        with torch.no_grad():
            # Maschera per stati non-terminali (True se next_state non è None), sul device corretto
            non_final_mask = torch.tensor(
                [ns is not None for ns in batch.next_state],
                dtype=torch.bool,
                device=self.device
            )

            # Costruisci tensore dei next_state solo per quelli non terminali
            if non_final_mask.any():
                non_final_next_states = torch.tensor(
                    np.stack([ns for ns in batch.next_state if ns is not None]),
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                # placeholder vuoto con shape (0, state_dim) se non ci sono next_state non-term
                non_final_next_states = torch.empty((0, ) + states.shape[1:], dtype=torch.float32, device=self.device)

            # Inizializza next_q_all a zero con lo stesso dtype/device di rewards (shape (B,1))
            next_q_all = torch.zeros((len(batch.state), 1), dtype=rewards.dtype, device=self.device)

            # Se esistono next_state non terminali, calcola il massimo Q dalla target network
            if non_final_next_states.numel() > 0:
                # target_net(non_final_next_states) -> output shape (N, num_actions)
                # .max(1)[0] prende il valore massimo per riga; poi unsqueeze(1) per avere shape (N,1)
                next_q_vals = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1)
                # Metti i valori calcolati nelle posizioni corrispondenti del batch usando la maschera
                next_q_all[non_final_mask] = next_q_vals

            # Target: reward + gamma * next_q_all per stati non-terminali; per terminali next_q_all è 0
            target = rewards + (1.0 - dones) * self.gamma * next_q_all  # shape (B,1)

        # Calcolo della loss tra Q corrente e target
        loss = self.loss_fn(q_values, target)

        # Azzeramento dei gradienti (set_to_none=True leggermente più efficiente)
        self.optimizer.zero_grad(set_to_none=True)

        # Backprop
        loss.backward()

        # Clipping dei gradienti per stabilità (opzionale, aggiustare max_norm se necessario)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)

        # Aggiorna i pesi della rete policy
        self.optimizer.step()

        # Incrementa il contatore di passi e aggiorna la rete target a intervalli prefissati
        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            # Copia i pesi della policy nella target
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decadimento dell'epsilon di esplorazione fino al minimo definito
        if self.explorationRate > self.explorationRate_min:
            self.explorationRate *= self.explorationRate_decay

#--------------------------------------------------------------------------------------------------------------------
    def save(self, file):
        # Salva gli stati delle reti, dell'ottimizzatore e metadati utili per ripristinare l'allenamento
        os.makedirs("models", exist_ok=True)
        path = f"models/{file}-{self.game.nrows}-{self.game.ncols}-{self.game.nTris}.pth"
        torch.save({
            'policy': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'explorationRate': self.explorationRate,
            'step_count': self.step_count,
        }, path)

#--------------------------------------------------------------------------------------------------------------------
    def load(self, file):
        path = f"models/{file}-{self.game.nrows}-{self.game.ncols}-{self.game.nTris}.pth"
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.target_net.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])


###############################################################################################################################################
# Training loop: integra con la classe Tris presente in tris.py
def train_dqn(agent, num_episodes, opponent='random'):

    for ep in range(1, num_episodes + 1):
        agent.game.re_init()
        state = board_to_tensor( agent.game, agent_mark=1)

        #stampa Prob prima mossa
        agent.print_action_rprobs(state, agent.game.available_actions())

#-----------------------------------------------------------------------------------------------------------
        while (not agent.game.game_over) and (bool(agent.game.available_actions())):

            state = board_to_tensor(agent.game, agent_mark=1)
            done = False
            next_state = None
            reward = 0
            
            if agent.game.current_player == agent.game.players[0]:

                action = agent.select_action(state, agent.game.available_actions())
                agent.game.make_move_action(action)

                if agent.game.winner == 'X':
                    reward = 1.0
                    done = True
                elif not agent.game.available_actions():
                    reward = 0.0                   
                    done = True
                else:
                    reward = 0.0
                    next_state = board_to_tensor(agent.game, agent_mark=1)

                agent.remember(state, action, reward, next_state, agent.game.run, done)
                agent.optimize()
                if done:
                    break

            else:
                opp_action = agent.game.calculate_action(0.0)
                agent.game.make_move_action(opp_action)
                if agent.game.winner == 'O':
                    reward = -1.0
                    done = True
                elif not agent.game.available_actions():
                    reward = 0.0
                    done = True
                else:
                    reward = 0.0
                
                if done:
                    agent.remember(state, action, reward, None, agent.game.run, done)
                    agent.optimize()
                    break

#-----------------------------------------------------------------------------------------------------------
        if ep % 500 == 0:
            print("")
            print(f"Episode {ep}, explorationRate {agent.explorationRate:.3f}, buffer {len(agent.replay)}")

            wins, draws, losses = test_dqn(agent, 1000)
            # Calcola e stampa i risultati
            total = wins + draws + losses
            accuracy = (wins / total) * 100 if total > 0 else 0
            print(f"  Vittorie:  {wins:3d} ({wins/total*100:5.1f}%)")
            print(f"  Pareggi:   {draws:3d} ({draws/total*100:5.1f}%)")
            print(f"  Sconfitte: {losses:3d} ({losses/total*100:5.1f}%)")
            print("")

###############################################################################################################################################
# train di agnet con agente opponente agentO
def train_dqn_agentO(agent, agentO, num_episodes, opponent='random'):

    for ep in range(1, num_episodes + 1):
        agent.game.re_init()
        state = board_to_tensor( agent.game, agent_mark=1)

        #stampa Prob prima mossa
        agent.print_action_rprobs(state, agent.game.available_actions())

#-----------------------------------------------------------------------------------------------------------
        while (not agent.game.game_over) and (bool(agent.game.available_actions())):

            state = board_to_tensor(agent.game, agent_mark=1)
            done = False
            next_state = None
            reward = 0
            
            if agent.game.current_player == agent.game.players[0]:

                action = agent.select_action(state, agent.game.available_actions())
                agent.game.make_move_action(action)

                if agent.game.winner == 'X':
                    reward = 1.0
                    done = True
                elif not agent.game.available_actions():
                    reward = 0.0                   
                    done = True
                else:
                    reward = 0.0
                    next_state = board_to_tensor(agent.game, agent_mark=1)

                agent.remember(state, action, reward, next_state, agent.game.run, done)
                agent.optimize()
                if done:
                    break

            else:
                opp_action = agentO.select_action(state, agent.game.available_actions() )
                agent.game.make_move_action(opp_action)
                if agent.game.winner == 'O':
                    reward = -1.0
                    done = True
                elif not agent.game.available_actions():
                    reward = 0.0
                    done = True
                else:
                    reward = 0.0
                
                if done:
                    agent.remember(state, action, reward, None, agent.game.run, done)
                    agent.optimize()
                    break

#-----------------------------------------------------------------------------------------------------------
        if ep % 500 == 0:
            print("")
            print(f"Episode {ep}, explorationRateX {agent.explorationRate:.3f}, buffer {len(agent.replay)}, explorationRateO {agentO.explorationRate:.3f}")

            wins, draws, losses = test_dqn(agent, 1000)
            # Calcola e stampa i risultati
            total = wins + draws + losses
            accuracy = (wins / total) * 100 if total > 0 else 0
            print(f"  Vittorie:  {wins:3d} ({wins/total*100:5.1f}%)")
            print(f"  Pareggi:   {draws:3d} ({draws/total*100:5.1f}%)")
            print(f"  Sconfitte: {losses:3d} ({losses/total*100:5.1f}%)")
            print("")

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
        agent.game.re_init()  

        while (not agent.game.game_over) and (bool(agent.game.available_actions())):

            if agent.game.current_player == agent.game.players[0]:           # Agent (X)
                avail = agent.game.available_actions()
                state = board_to_tensor(agent.game, agent_mark=1)
                action = agent.select_action(state, avail)
                agent.game.make_move_action(action)
            else:  # Opponent (O) - random
                opp_action = agent.game.random_action()       # Player (O)
                agent.game.make_move_action(opp_action)

        # Count result
        if agent.game.winner == 'X':
            wins += 1
        elif agent.game.winner == 'O':
            losses += 1
        else:
            draws += 1

    print(f"Partite completate: {game_idx + 1}/{num_games}")
    return wins, draws, losses

###############################################################################################################################################
def test_match_dqn( agentX, agentO, num_games=100):
    """
    Test l'agente DQN contro un avversario random per num_games partite.
    Ritorna il numero di vittorie, pareggi e sconfitte.
    """
    wins = 0
    draws = 0
    losses = 0

    for game_idx in range(num_games):
        agentX.game.re_init()                 #agentX.game == agentO.game 

        while (not agentX.game.game_over) and (bool(agentX.game.available_actions())):

            if agentX.game.current_player == agentX.game.players[0]:           # Agent (X)
                avail = agentX.game.available_actions()
                state = board_to_tensor(agentX.game, agent_mark=1)
                action = agentX.select_action(state, avail)
                agentX.game.make_move_action(action,1000)
            else:  # Opponent Agent (O) la board e' in agentX
                avail = agentO.game.available_actions()
                state = board_to_tensor(agentO.game, agent_mark=1)
                action = agentO.select_action(state, avail)                     # << Uso agentO
                agentO.game.make_move_action(action, 1000)

        # Count result agentX.game == agentO.game
        if agentX.game.winner == 'X':
            wins += 1
        elif agentX.game.winner == 'O':
            losses += 1
        else:
            draws += 1

    print(f"Partite completate: {game_idx + 1}/{num_games}")
    return wins, draws, losses
