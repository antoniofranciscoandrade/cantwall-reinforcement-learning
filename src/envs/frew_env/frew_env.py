import gym
from .frew import Frew
import numpy as np
import time

class FrewEnv(gym.Env):

    def __init__(self, wall_depth_bounds, pile_d, max_deflection):

        self.frw = Frew(r"C:\projects\frw-rl\models\CANTTEST.fwd")

        self.max_deflection = max_deflection
        self.action_space = []
        self.generate_action_space(wall_depth_bounds, pile_d)
        self.step_counter = 0
        self.max_structural_reward = 0

    def generate_action_space(self, wall_depth_bounds, pile_d):

        self.minimum_volume = np.pi*(min(pile_d)**2)*wall_depth_bounds[0]/4
        self.maximum_volume = np.pi*(max(pile_d)**2)*wall_depth_bounds[1]/4

        depths = np.arange(wall_depth_bounds[0], wall_depth_bounds[1], wall_depth_bounds[2])
        
        for depth in depths:
            for diameter in pile_d:
                pile_spacings = np.arange(diameter, 2*diameter + 0.1, 0.1)
                for spacing in pile_spacings:
                    self.action_space.append([depth, diameter, spacing])

    def compute_reward(self, sim_results, wall_depth, pile_diameter):
        # add steel consumption

        concrete_volume = np.pi*(pile_diameter**2)*wall_depth/4
        concrete_volume_norm = ((concrete_volume - self.minimum_volume)/(self.maximum_volume - self.minimum_volume))
        concrete_reward = 1/max(0.01, concrete_volume_norm)
        
        deflection_reward = 1 - (sim_results['max_deflection']/self.max_deflection)**0.4

        structural_reward = deflection_reward*concrete_reward

        # step_reward = structural_reward*(1-1/(1+self.step_counter))

        if self.step_counter == 1:
            self.max_structural_reward = structural_reward

        greedy_curiosity = 1
        if structural_reward > self.max_structural_reward:
            greedy_curiosity = structural_reward/self.max_structural_reward
            self.max_structural_reward = structural_reward
            
        return greedy_curiosity*structural_reward, structural_reward
                 
    def step(self, action):

        self.step_counter += 1 
        wall_depth, pile_diameter, pile_spacing = \
                                    self.action_space[action]

        sim_results = self.frw.cantilever_analysis(
                                    wall_depth,
                                    pile_diameter,
                                    pile_spacing
                                    )

        reward, structural_reward = self.compute_reward(sim_results, wall_depth, pile_diameter)
        new_state = sim_results['max_deflection']

        done = False
        if sim_results['max_deflection'] >= self.max_deflection:
            done = True
            self.step_counter = 0

        return (new_state, reward, structural_reward, done)

    def reset(self):
        
        self.frw.reset()
        self.step_counter = 0

        return 1e5


