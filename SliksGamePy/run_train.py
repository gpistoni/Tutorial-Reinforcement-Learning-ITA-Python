from dqn_agent import DQNAgent, train_dqn
from sliks import DrivingGame

###############################################################################################################################################
if __name__ == "__main__":
    #import gym
    import numpy as np
    from argparse import ArgumentParser

    # --- Crea ambiente ---
    game = DrivingGame(fileMap="SliksGamePy/track.png", render_decmation=10, fps=1000 )

    # --- Crea agente ---
    agent = DQNAgent(
        input_dim = game.getState_dim(),
        output_dim = game.getAction_dim(),
        lr=1e-4,
        tau=1e-2,
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
        model_path="models/dqn_slicks.pth",
    )

    # Salva modello finale
    agent.save("models/dqn_final.pth")

    # --- Test ---
    #print("Testing (greedy)...")
    #results = test_dqn(agent, game, num_episodes=args.test_episodes, max_steps_per_episode=500, render=False)
    #print(f"Test rewards per episode: {results}")
    #print(f"Average test reward: {sum(results)/len(results):.2f}")
