import random
import torch
from dqn_agent import DQNAgent, test_dqn
from tris import Tris


###############################################################################################################################################
if __name__ == '__main__':

    num_games = 1000

    game = Tris(3,3,3)

    print("=" * 60)
    print("Test RANDOM Agent - Tris")
    print("=" * 60)

    # Esegui il test su 100 partite
    print("-" * 60)
    agent = DQNAgent(device='cuda', game= game, explorationRate = 1 )
    wins0, draws0, losses0 = test_dqn(agent, num_games)
    print("-" * 60)

    print("=" * 60)
    print("Test DQN Agent - Tris")
    print("=" * 60)

    # Carica l'agente e il modello salvato
    agent = DQNAgent(device='cuda', game=game, explorationRate = 0 )
    try:
        agent.load('dqn_game')
        print("✓ Modello 'dqn_tris_final.pth' caricato con successo")
    except FileNotFoundError:
        print("✗ Modello non trovato. Assicurati di aver eseguito 'run_train.py' prima.")
        exit(1)

    # Esegui il test su 100 partite
    print("-" * 60)
    wins, draws, losses = test_dqn(agent, num_games)
    print("-" * 60)


    print("\nRISULTATI RND:")
    print(f"  Vittorie:  {wins0:3d} ({wins0/num_games*100:5.1f}%)")
    print(f"  Pareggi:   {draws0:3d} ({draws0/num_games*100:5.1f}%)")
    print(f"  Sconfitte: {losses0:3d} ({losses0/num_games*100:5.1f}%)")
    print("=" * 60)

    print("\nRISULTATI:")
    print(f"  Vittorie:  {wins:3d} ({wins/num_games*100:5.1f}%)")
    print(f"  Pareggi:   {draws:3d} ({draws/num_games*100:5.1f}%)")
    print(f"  Sconfitte: {losses:3d} ({losses/num_games*100:5.1f}%)")
    print("=" * 60)
