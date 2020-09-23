from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from envs.frew_env import FrewEnv
import time
import pandas as pd
import numpy as np


def build_dqn(lr, n_actions, input_dim, fcl_dim):

    model = Sequential()
    model.add(Dense(fcl_dim[0], input_dim=input_dim, activation='relu'))
    for shapes in fcl_dim[1:]:
        model.add(Dense(shapes, activation='relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model

def load_dqn(weights_path, lr, n_actions, input_dim, fcl_dim):
   model = build_dqn(lr, n_actions, input_dim, fcl_dim)
   model.load_weights(weights_path)

   return model

def generate_log(env, q):

    action_space_df = pd.DataFrame(env.action_space, columns=['depth', 'diameter', 'spacing'])

    for i, row in action_space_df.iterrows():
        action_space_df.loc[i, 'reward'] = env.step(i)[2]
        action_space_df.loc[i, 'q'] = q[i]

    action_space_df = action_space_df.sort_values('reward', ascending=False)
    action_space_df.to_csv('output/actions_rewards_q.csv')
    action_space_df = action_space_df.sort_values('q', ascending=False)
    action_space_df.to_csv('output/actions_q_sorted.csv')

if __name__ == '__main__':

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

    n_actions=len(env.action_space)

    dqn = load_dqn(
        '../models/dqn_model.h5',
        lr=0.01,
        n_actions=n_actions,
        input_dim=1,
        fcl_dim=[256, 256, 256, 256]
    )

    results = pd.read_csv('output/all_rewards.csv')

    start = time.time()
    print('started')
    q = dqn.predict([1e5])
    print(f'{(time.time() - start)} seconds')

    start = time.time()
    generate_log(env, q.flatten())
    print(f'{(time.time() - start)/60} minutes')