from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np

class ReplayMemory(object):

    def __init__(self, max_size, input_shape, n_actions):

        self.mem_counter = 0
        self.mem_size = int(max_size)
        self.input_shape = input_shape
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros(self.mem_size, dtype=int)
        self.reward_memory = np.zeros(self.mem_size)
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.terminal_memory = np.zeros(self.mem_size, dtype=float)

    def store_transition(self, state, action, reward, new_state, done):

        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.action_memory[index] = action
        self.mem_counter += 1

    def sample_memory(self, batch_size):

        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal 


def build_dqn(lr, n_actions, input_dim, fcl_dim):

    model = Sequential()
    model.add(Dense(fcl_dim[0], input_dim=input_dim, activation='relu'))
    for shapes in fcl_dim[1:]:
        model.add(Dense(shapes, activation='relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model


class Agent(object):
    
    def __init__(self, alpha, gamma, n_actions, eps, batch_size, 
                 input_dim, eps_dec=0.996, eps_end=0.01, nn_arch=[256, 256],
                 mem_size=1e6, fname='../models/dqn_model.h5'):

        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayMemory(mem_size, input_dim, n_actions)

        self.q_eval = build_dqn(alpha, n_actions, input_dim, nn_arch)

    def remember(self, state, action, reward, new_state, done):

        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        
        if type(state) != np.array:
            state = np.array([state])

        state = state[np.newaxis, :]
        rand = np.random.random()

        if rand < self.eps:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):

        if self.memory.mem_counter < self.batch_size:
            return
        
        state, action, reward, new_state, done = \
                                   self.memory.sample_memory(self.batch_size)
        
        
        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=int)

        q_target[batch_index, action] = reward + self.gamma*np.max(q_next, axis=1)*(1-done)

        _ = self.q_eval.fit(state, q_target, verbose=0)

        self.eps = self.eps*self.eps_dec if self.eps > self.eps_min else self.eps_min
    
    def latest_successful_action(self):
        latest_index = self.memory.mem_counter%self.memory.mem_size - 1
        
        if self.memory.mem_counter < self.batch_size:
            return

        return self.memory.action_memory[latest_index-1]

    def save_model(self):
        self.q_eval.save(self.model_file)
    
    def load_model(self):
        self.q_eval = load_model(self.model_file)


