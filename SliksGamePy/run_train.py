from dqn_agent import DQNAgent, train_dqn
from sliks import DrivingGame

###############################################################################################################################################
if __name__ == "__main__":
    #import gym
    import numpy as np
    from argparse import ArgumentParser

    # --- Crea ambiente ---
    game = DrivingGame(fileMap="SliksGamePy/track_0.png", render_decmation=8, fps=100 )

    """
    # fallback: env mock con obs dim 16 e 4 azioni per testing rapido
    class DummyEnv:
        def __init__(self):
            self.observation_space = type("S", (), {"shape": (16,)})
            self.action_space = type("A", (), {"n": 4})
        def reset(self):
            return np.zeros(16, dtype=np.float32)
        def step(self, action):
            next_s = np.random.randn(16).astype(np.float32)
            reward = float(np.random.rand() * 10000.0)  # reward in [0,10000)
            done = np.random.rand() < 0.01
            return next_s, reward, done, {}
        def render(self): pass
    game = DummyEnv()
    """

    # --- Crea agente ---
    agent = DQNAgent(
        input_dim = game.getState_dim(),
        output_dim = game.getAction_dim(),
        lr=1e-3,
        batch_size=128,
        buffer_capacity=50000,
        target_update=1000,
        eps_decay_steps=50000,
        hidden_sizes=[128, 128],
    )

    # --- Train ---
    print("Training...")
    train_dqn(
        agent,
        game,
        num_episodes = 1000,
        max_steps_per_episode = 2000,
        reward_scale = 1.0,      
        model_path="models/dqn_slicks.pth",
    )

    # Salva modello finale
    agent.save("models/dqn_final.pth")

    # --- Test ---
    #print("Testing (greedy)...")
    #results = test_dqn(agent, game, num_episodes=args.test_episodes, max_steps_per_episode=500, render=False)
    #print(f"Test rewards per episode: {results}")
    #print(f"Average test reward: {sum(results)/len(results):.2f}")
