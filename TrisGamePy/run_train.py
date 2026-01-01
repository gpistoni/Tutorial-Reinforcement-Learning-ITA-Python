from dqn_agent import DQNAgent, train_dqn

###############################################################################################################################################
if __name__ == '__main__':
    a = DQNAgent()
    train_dqn(a, num_episodes=20000)
    a.save('dqn_tris_final.pth')

