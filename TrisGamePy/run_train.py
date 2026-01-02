from dqn_agent import DQNAgent, train_dqn

###############################################################################################################################################
if __name__ == '__main__':
    num_episodes = 5000
    a = DQNAgent()
    train_dqn(a, num_episodes)
    a.save('dqn_tris_final.pth')

