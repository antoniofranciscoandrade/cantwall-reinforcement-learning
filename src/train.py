from rl import Agent
import numpy as numpy
import gym
from envs.frew_env import FrewEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time


if __name__ == '__main__':

    # standardized pile diameters
    pile_diameters = [
        0.2, 
        0.25,
        0.45,
        0.6,
        0.8,
        0.9,
        1.2,
        1.5,
        1.8,
        2.1,
        2.15,
        2.2,
        2.25,
        2.3,
        2.35,
        2.4,
        2.45,
        2.5,
        2.55,
        2.6,
        2.65,
        2.7
    ]

    env = FrewEnv(
            wall_depth_bounds=[5, 45, 1], 
            pile_d=pile_diameters, 
            max_deflection=1000
    )

    n_episodes = int(1.5e3)
    n_actions=len(env.action_space)

    agent = Agent(
        alpha=0.01, 
        gamma=0.9,  
        eps=1, 
        batch_size=256, 
        n_actions=n_actions,
        input_dim=1,
        eps_dec=0.998,
        nn_arch=[256, 256, 256, 256]
    )

    scores = []
    normalized_rewards = []
    eps_history = []
    steps = 0
    total_steps = 0

    fig = plt.figure(figsize=(7,7))
    plt.get_current_fig_manager().window.wm_geometry("+10+10")
    steps_ax = fig.add_subplot(2, 1, 1)
    episode_ax = fig.add_subplot(2, 1, 2)

    steps_ax.set_xlabel('step')
    steps_ax.set_ylabel('structural reward')

    episode_ax.set_xlabel('episode')
    episode_ax.set_ylabel('average structural reward')
    
    fig.show()
    fig.canvas.draw()
    try:
        start = time.time()
        for i in range(n_episodes):

            done = False
            score = 0
            steps = 0
            state = env.reset()

            while not done:
                action = agent.choose_action(state)
                new_state, reward, structural_reward, done = env.step(action)
                score += structural_reward
                agent.remember(state, action, reward, new_state, done)
                state = new_state
                agent.learn()
                steps+=1
                total_steps += 1
                normalized_rewards.append(structural_reward)
                steps_ax.plot(np.arange(total_steps), normalized_rewards)
                fig.canvas.draw()

            eps_history.append(agent.eps)
            avg_score = score/steps
            scores.append(avg_score)
            
            episode_ax.plot(np.arange(len(scores)), scores)
            steps_ax.axvline(x=total_steps-1, color='grey',linestyle='--')
            fig.canvas.draw()

            latest_action = agent.latest_successful_action()
            action_value = []
            if latest_action:
                action_value = env.action_space[latest_action]

            print(f'episode {i} score {score} average score {avg_score} steps {steps} epsilon {agent.eps} action {action_value}')

            if i % 10 == 0:
                agent.save_model()
        
    except KeyboardInterrupt:
        print(f'Total number of steps: {total_steps}')
        plt.pause(0.01)
        input("<Hit Enter To Close>")
        latest_action = agent.latest_successful_action()
        print(f'episode {i} score {score} average score {avg_score} steps {steps} epsilon {agent.eps} action {action_value}')
        print(f'{(time.time() - start)/60} minutes')
        agent.save_model()
        sys.exit(0)

    plt.pause(0.01)
    input("<Hit Enter To Close>")
    print(f'Total number of steps: {total_steps}')
    print(f'{(time.time() - start)/60} minutes')



