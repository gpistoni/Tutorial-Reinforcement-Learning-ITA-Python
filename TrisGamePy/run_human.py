import random
import torch
from dqn_agent import DQNAgent, test_human_dqn
from tris import Tris


###############################################################################################################################################
if __name__ == '__main__':

    num_games = 10

    game = Tris(3,3,3)

    print("=" * 60)
    print("Test HUMAN VS Agent - Tris")
    print("=" * 60)

    # Esegui il test su 100 partite
    agentX = DQNAgent(device='cuda', game = game, explorationRate = 0 )
    agentX.load('dqn_game_match')
    wins, draws, losses = test_human_dqn(agentX, num_games)
    print("-" * 60)

    print("\nRISULTATI MATCH:")
    print(f"  Vittorie X:  {wins:3d} ({wins/num_games*100:5.1f}%)")
    print(f"  Pareggi:   {draws:3d} ({draws/num_games*100:5.1f}%)")
    print(f"  Sconfitte: {losses:3d} ({losses/num_games*100:5.1f}%)")
    print("=" * 60)
