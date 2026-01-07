import random
import torch
from dqn_agent import DQNAgent, test_dqn
from sliks import DrivingGame


###############################################################################################################################################
if __name__ == '__main__':

    # --- Crea ambiente ---
    game = DrivingGame(fileMap="SliksGamePy/track_0.png", render_decmation=1, fps=30, max_speed=4.0 )

    # --- Crea agente ---
    agent = DQNAgent(
        input_dim = game.getState_dim(),
        output_dim = game.getAction_dim(),
        lr=1e-3,
        tau=1e-1,
        batch_size=128,
        buffer_capacity=100000,
        target_update=1000,
        eps_decay_steps=50000,
        hidden_sizes=[128, 128],
    )

    # --- Train ---
    print("Test...")
    test_dqn(agent, game, 
                num_episodes = 10, 
                max_steps_per_episode = 1000, 
                model_path="models/dqn_slicks.pth",
                )