from dqn_agent import DQNAgent, train_dqn, train_dqn_agentO
from tris import Tris, Action

###############################################################################################################################################
if __name__ == '__main__':

    num_episodes = 3000
    game=Tris(3,3,3)

    if False:
        agent = DQNAgent(device='cuda', game=game, explorationRate=1.0 )
        train_dqn(agent, num_episodes)
        agent.save('dqn_game')
    else:
        agent = DQNAgent(device='cuda', game=game, explorationRate=1.0 )

        agentQ = DQNAgent(device='cuda', game=game, explorationRate=0 )
        agentQ.load('dqn_game')

        train_dqn_agentO(agent, agentQ, num_episodes)
        agent.save('dqn_game_match')
