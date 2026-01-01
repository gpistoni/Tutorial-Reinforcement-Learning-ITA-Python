import random
import torch
from dqn_agent import DQNAgent, board_to_tensor
from tris import Tris

###############################################################################################################################################
def test_agent(agent, num_games=100):
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
        game.current_player = random.choice(game.players)  # random chi inizia
        state = board_to_tensor(game.board, agent_mark=1)

        while (not game.game_over) and (bool(game.available_moves())):
            if agent and game.current_player == game.players[0]:      # Agent (X)
                avail = game.available_moves()
                action = agent.select_action(state, avail)
                if not game.make_move(action):
                    action = random.choice(avail)
                    game.make_move(action)
            else:  # Opponent (O) - random
                opp_action = random.choice(game.available_moves())       # Player (O)
                game.make_move(opp_action)

            # Update state after opponent's move
            if not game.game_over and bool(game.available_moves()):
                state = board_to_tensor(game.board, agent_mark=1)

        # Count result
        if game.winner == 'X':
            wins += 1
        elif game.winner == 'O':
            losses += 1
        else:
            draws += 1

        if (game_idx + 1) % 20 == 0:
            print(f"  Partite completate: {game_idx + 1}/{num_games}")

    return wins, draws, losses

###############################################################################################################################################
if __name__ == '__main__':

    num_games = 1000

    print("=" * 60)
    print("Test RANDOM Agent - Tris")
    print("=" * 60)

    # Esegui il test su 100 partite
    print("\nEsecuzione test su 100 partite...")
    print("-" * 60)
    wins0, draws0, losses0 = test_agent(None, num_games)
    print("-" * 60)

    # Calcola e stampa i risultati
    total0 = wins0 + draws0 + losses0
    accuracy0 = (wins0 / total0) * 100 if total0 > 0 else 0

    print("=" * 60)
    print("Test DQN Agent - Tris")
    print("=" * 60)

    # Carica l'agente e il modello salvato
    agent = DQNAgent(device='cpu')
    try:
        agent.load('dqn_tris_final.pth')
        print("✓ Modello 'dqn_tris_final.pth' caricato con successo")
    except FileNotFoundError:
        print("✗ Modello non trovato. Assicurati di aver eseguito 'run_train.py' prima.")
        exit(1)

    # Esegui il test su 100 partite
    print("\nEsecuzione test su 100 partite...")
    print("-" * 60)
    wins, draws, losses = test_agent(agent, num_games)
    print("-" * 60)

    # Calcola e stampa i risultati
    total = wins + draws + losses
    accuracy = (wins / total) * 100 if total > 0 else 0

    print("\nRISULTATI RND:")
    print(f"  Vittorie:  {wins0:3d} ({wins0/total*100:5.1f}%)")
    print(f"  Pareggi:   {draws0:3d} ({draws0/total*100:5.1f}%)")
    print(f"  Sconfitte: {losses0:3d} ({losses0/total*100:5.1f}%)")
    print(f"  ACCURATEZZA: {accuracy0:.1f}%")
    print("=" * 60)

    print("\nRISULTATI:")
    print(f"  Vittorie:  {wins:3d} ({wins/total*100:5.1f}%)")
    print(f"  Pareggi:   {draws:3d} ({draws/total*100:5.1f}%)")
    print(f"  Sconfitte: {losses:3d} ({losses/total*100:5.1f}%)")
    print(f"  ACCURATEZZA: {accuracy:.1f}%")
    print("=" * 60)
