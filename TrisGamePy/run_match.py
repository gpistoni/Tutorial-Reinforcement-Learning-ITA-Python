import random
import torch
from dqn_agent import DQNAgent, test_dqn, test_match_dqn
from tris import Tris


###############################################################################################################################################
if __name__ == '__main__':

    num_games = 1000

    game = Tris(3,3,3)

    print("=" * 60)
    print("Test MATCH Agent - Tris")
    print("=" * 60)

    # Esegui il test su 100 partite
    agentX = DQNAgent(device='cuda', game = game, explorationRate = 0 )
    agentX.load('dqn_game')
    agentO = DQNAgent(device='cuda', game = game, explorationRate = 0 )
    agentO.load('dqn_game_adv')
    wins, draws, losses = test_match_dqn(agentX, agentO, num_games)
    print("-" * 60)

    print("\nRISULTATI MATCH:")
    print(f"  Vittorie:  {wins:3d} ({wins/num_games*100:5.1f}%)")
    print(f"  Pareggi:   {draws:3d} ({draws/num_games*100:5.1f}%)")
    print(f"  Sconfitte: {losses:3d} ({losses/num_games*100:5.1f}%)")
    print("=" * 60)
