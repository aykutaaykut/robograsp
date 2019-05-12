#!/usr/bin/env python

from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
import gym

class Agent:
	def __init__(self, state_dim, action_dim, memory_size, lr, discount, epsilon, epsilon_decay, epsilon_min, batch_size):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.memory = deque(maxlen = memory_size)
		self.lr = lr
		self.discount = discount
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.batch_size = batch_size
		self.network = self.create_dl_network(input_dim = self.state_dim, output_dim = self.action_dim, optimizer_lr = self.lr)

	def create_dl_network(self, input_dim, output_dim, optimizer_lr):
		network = Sequential()
#		network.add(Dense(24, input_dim = input_dim, kernel_initializer = 'RandomUniform', activation = 'elu'))
#		network.add(Dense(64, kernel_initializer = 'RandomUniform', activation = 'elu'))
#		network.add(Dense(256, kernel_initializer = 'RandomUniform', activation = 'elu'))
#		network.add(Dense(512, kernel_initializer = 'RandomUniform', activation = 'elu'))
#		network.add(Dense(1024, kernel_initializer = 'RandomUniform', activation = 'elu'))
		network.add(Dense(output_dim, activation = 'linear'))
		network.compile(loss = 'mse', optimizer = Adam(lr = optimizer_lr))
		return network

	def save2memory(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def choose_action(self, state):
	    return self.network.predict(state)[0]

	def policy(self, state):
		toss = np.random.rand()
		if toss <= self.epsilon:
			return random.randrange(self.action_dim)
		else:
			return np.argmax(self.choose_action(state))

	def experience_replay(self):
		batch = random.sample(self.memory, self.batch_size)
		for s, a, r, n_s, d in batch:
			q_target_val = r
			if not d:
				q_target_val = q_target_val + (self.discount * np.amax(self.choose_action(s)))
			q_target = self.choose_action(s)
			q_target[a] = q_target_val
			self.network.fit(s, q_target.reshape(1, self.action_dim), epochs = 1, verbose = 0)

	def decay_epsilon(self):
	    if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load_network(self, file_path):
		self.network = load_model(file_path)

	def save_network(self, file_path):
		return self.network.save(file_path)

	def self_print(self):
	    return '#state_dim: ' + str(self.state_dim) + '\n#action_dim: ' + str(self.action_dim) + '\n#memory_size: ' + str(len(self.memory)) + '\n#lr: ' + str(self.lr) + '\n#discount: ' + str(self.discount) + '\n#epsilon: ' + str(self.epsilon) + '\n#epsilon_decay: ' + str(self.epsilon_decay) + '\n#epsilon_min: ' + str(self.epsilon_min) + '\n#batch_size: ' + str(self.batch_size)
