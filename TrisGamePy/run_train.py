from dqn_agent import DQNAgent, train_dqn
from tris import Tris, Action

###############################################################################################################################################
if __name__ == '__main__':
    num_episodes = 2000

    agent = DQNAgent(device='cuda', game=Tris(3,3,3), explorationRate=1.0 )
    train_dqn(agent, num_episodes)
    agent.save('dqn_game')

