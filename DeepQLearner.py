import random

import numpy as np
import tensorflow as tf
import pandas as pd

class DeepQLearner:
    def __init__ (self, state_dim = 3, action_dim = 3, alpha = 0.2, gamma = 0.9, epsilon = 0.98,
                  epsilon_decay = 0.999, hidden_sizes = (32, 32)):
        # Store all the parameters as attributes (instance variables).
        # Initialize any data structures you need.
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.experiences = []
        self.hidden_sizes = hidden_sizes
        self.prev_s = [0 for _ in range(state_dim)]
        self.prev_a = 0

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.hidden_sizes[0], input_dim=self.state_dim, activation='relu'))

        for layer_size in self.hidden_sizes:
            model.add(tf.keras.layers.Dense(layer_size, activation='relu'))

        model.add(tf.keras.layers.Dense(self.action_dim, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),
                      loss='mse')
        return model

    def loss_function(self, r, exp_future_rewards, old_est):
        return (r + self.gamma * exp_future_rewards - old_est)**2

    def train(self, s, r):
        # Receive new state s and new reward r.  Update Q-table and return selected action.
        # Consider: The Q-update requires a complete <s, a, s', r> tuple.
        #           How will you know the previous state and action?
        prev_state = np.array(self.prev_s)
        current_state = np.array(s)
        prev_q_values = self.model.predict(np.array(prev_state))
        q_vals = self.model.predict(current_state)
        a = np.argmax(q_vals)

        target_q = r + self.gamma * q_vals[a]
        prev_q_values[self.prev_a] = target_q

        self.model.fit(np.array(prev_state), prev_q_values, verbose=0)

        self.experiences.append((self.prev_s, self.prev_a, s, r))
        self.prev_s = s
        self.prev_a = a
        self.epsilon *= self.epsilon_decay
        return a

    def test(self, s):
        # Receive new state s.  Do NOT update Q-table, but still return selected action.
        #
        # This method is called for TWO reasons: (1) to use the policy after learning is finished, and
        # (2) when there is no previous state or action (and hence no Q-update to perform).
        #
        # When testing, you probably do not want to take random actions... (What good would it do?)
        q_vals = self.model.predict(np.array(s))
        a = np.argmax(q_vals)
        self.prev_s = s
        self.prev_a = a

        return a

    def update(self, alpha, gamma, r, previous, exp_future_rewards):
        return (1 - alpha) * previous + alpha * (r + gamma * exp_future_rewards)

    def choose_action(self, s):
        if random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_vals = self.model.predict(np.array(s))
            return np.argmax(q_vals)