from dqn_agent import DQNAgent, train_dqn, train_dqn_agentO
from sliks import DrivingGame

###############################################################################################################################################
if __name__ == '__main__':

    num_episodes = 100
    lr = 1e-3
    game = DrivingGame()

    if True:
        agent = DQNAgent(device='cuda', game=game, explorationRate=1.0 )
        train_dqn(agent, num_episodes)
        agent.save('dqn_game')

    else:
        agent = DQNAgent(device='cuda', game=game, explorationRate=1.0, lr = lr )

        agentQ = DQNAgent(device='cuda', game=game, explorationRate=0.4, lr = lr )
        agentQ.load('dqn_game')

        train_dqn_agentO(agent, agentQ, num_episodes)
        agent.save('dqn_game_match')
