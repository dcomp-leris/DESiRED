import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from numpy.random import seed
seed(1)

import tensorflow as tf
tf.random.set_seed(1)

import random
random.seed(1)

from collections import deque
from random import sample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2

from sklearn import preprocessing

class DDQN:
    def __init__(self, state_dim,
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 tau,
                 batch_size,
                 minimum_experience_memory,
                 initialization, # standard or pretrained
                 online_network_filepath,
                 target_network_filepath,
                 experiment_type):
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg
        self.minimum_experience_memory = minimum_experience_memory

        self.initialization = initialization
        self.online_network_filepath = online_network_filepath
        self.target_network_filepath = target_network_filepath

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.q_values = []
        self.idx = tf.range(batch_size)
        self.train = True

        self.experiment_type = experiment_type

        if self.initialization == 'standard':
            self.online_network = self.build_model()
            self.target_network = self.build_model(trainable=False)

            self.update_target()
        elif self.initialization == 'pretrained':
            self.load_models()


    def build_model(self, trainable=True):
        layers = []
        n = len(self.architecture)

        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
                                activation='relu',
                                #kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        
        layers.append(Dense(units=self.num_actions,
                            activation='linear',
                            trainable=trainable,
                            name='Output'))
        
        model = Sequential(layers)

        sgd = SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
        
        model.compile(loss='mean_squared_error',
                      optimizer=sgd)

        model.summary()
        
        return model
    

    def load_models(self):
        # if the experiment_type is 1, the q-network will be loaded and updated throughout the new experiment
        self.online_network = tf.keras.saving.load_model(self.online_network_filepath)
        self.online_network.summary()

        self.target_network = tf.keras.saving.load_model(self.target_network_filepath)
        self.target_network.summary()

        self.update_target()

        if self.experiment_type == 2:
            # if the experiment_type is 2, the q-network will only make predictions throughout the new experiment, not learning
            print("\n=================================================================\n")
            
            print("freezing the q-network layers weights (All layers):\n")
    
            # Iterating over the q-network layers
            for layer in self.online_network.layers:
                # Freezing the layers weights
                layer.trainable = False
                # Printing the layers 'trainable' parameter for sanity check
                print("{}: {}".format(layer.name, layer.trainable))
            
            print("\n=================================================================\n")
        
        if self.experiment_type == 3:
            # if the experiment_type is 3, apply the transfer learning from a pretrained q-network
            print("\n=================================================================\n")
            
            print("removing the pretrained q-network output layer...\n")

            self.online_network = tf.keras.Sequential(self.online_network.layers[:-2])

            print("freezing the q-network layers weights:\n")
    
            # Iterating over the q-network layers
            for layer in self.online_network.layers:
                # Freezing the layers weights
                layer.trainable = False
                # Printing the layers 'trainable' parameter for sanity check
                print("{}: {}".format(layer.name, layer.trainable))
            
            print("adding a new output layer...\n")

            self.online_network.add(Dense(units=self.num_actions,
                                       activation='linear',
                                       name='Output'))
            
            print("sanity check...\n")

            # Iterating over the model layers
            for layer in self.online_network.layers:
                # Printing the layers 'trainable' parameter for sanity check
                print("{}: {}".format(layer.name, layer.trainable))
            
            print("recompiling the model...\n")

            sgd = SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
        
            self.online_network.compile(loss='mean_squared_error',
                                     optimizer=sgd)

            print("\n=================================================================\n")
        
        #return online_q_network, target_q_network
    

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())
    

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        
        normalized_states = preprocessing.normalize(state)

        q = self.online_network.predict(normalized_states)
        
        return np.argmax(q, axis=1).squeeze()
    

    def memorize_transition(self, s, a, r, s_prime, not_done):
        
        if not_done == 0:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

        self.episodes += 1
        self.rewards_history.append(self.episode_reward)
        self.steps_per_episode.append(self.episode_length)
        self.episode_reward, self.episode_length = 0, 0

        self.experience.append((s, a, r, s_prime, not_done))
    

    def experience_replay(self):
        if self.minimum_experience_memory > len(self.experience):
            return

        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        normalized_states = preprocessing.normalize(states)
        normalized_next_states = preprocessing.normalize(next_states)

        next_q_values = self.online_network.predict_on_batch(normalized_next_states)
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_network.predict_on_batch(normalized_next_states)
        
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        targets = rewards + not_done * self.gamma * target_q_values

        self.q_values.append(np.mean(targets))

        q_values = self.online_network.predict_on_batch(normalized_states)

        
        q_values[self.idx, actions] = targets

        #print(states)
        #print("\n")
        #print(q_values)

        loss = self.online_network.train_on_batch(x=normalized_states, y=q_values)
        self.losses.append(loss)

        print("\n========================================\n", end='\r')
        print("learning from experience replay\n", end='\r')
        print("loss: {0} - mean q_value: {1}".format(loss, np.mean(targets)), end='\r')
        print("\n========================================\n", end='\r')

        if self.total_steps % self.tau == 0:
            self.update_target()
    

    def reset_metrics(self):
        self.losses = []
        self.q_values = []

    def save_agent(self, experiment_id):
        self.online_network.save('agent_models/online_network_{0}.h5'.format(experiment_id))
        self.target_network.save('agent_models/target_network_{0}.h5'.format(experiment_id))
