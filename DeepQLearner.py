import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class DeepQLearner:
    def __init__ (self, state_dim = 3, action_dim = 3, alpha = 0.9, gamma = 0.9, epsilon = 0.98,
                  epsilon_decay = 0.999, hidden_layers = 3, buffer_size = 100, batch_size = 32):
        # Store all the parameters as attributes (instance variables).
        # Initialize any data structures you need.
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience_buffer = []
        self.hidden_sizes = [state_dim*action_dim for _ in range(hidden_layers)]
        self.prev_s = [0 for _ in range(state_dim)]
        self.prev_a = 0

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.hidden_sizes[0], input_dim=self.state_dim, activation='relu'))

        for layer_size in self.hidden_sizes:
            model.add(tf.keras.layers.Dense(layer_size, activation='relu'))

        model.add(tf.keras.layers.Dense(self.action_dim, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.alpha),
                      loss='mse')
        return model

    def sample_from_buffer(self):
        if len(self.experience_buffer) < self.batch_size:
            return random.choices(self.experience_buffer, k=self.batch_size)
        else:
            return random.sample(self.experience_buffer, self.batch_size)


    def train(self, s, r):
        # Receive new state s and new reward r.  Update Q-table and return selected action.
        # Consider: The Q-update requires a complete <s, a, s', r> tuple.
        #           How will you know the previous state and action?
        current_state = np.array([s])
        q_vals = self.model.predict(current_state, verbose=0)
        print("Q-values in train:", q_vals)
        a = self.choose_action(s)

        self.experience_buffer.append((self.prev_s, self.prev_a, current_state, r))

        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer = self.experience_buffer[-self.buffer_size:]

        batch = self.sample_from_buffer()
        prev_states = [experience[0] for experience in batch]
        prev_actions = [experience[1] for experience in batch]
        next_states = [experience[2] for experience in batch]
        rewards = [experience[3] for experience in batch]

        prev_states = np.array(prev_states)
        prev_actions = np.array(prev_actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)

        prev_q_values = self.model.predict(prev_states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)
        targets = rewards + self.gamma * np.amax(max_next_q_values, axis=1)
        prev_q_values[np.arange(self.batch_size), prev_actions] = targets

        self.model.fit(prev_states, prev_q_values, verbose=0)

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
        q_vals = self.model.predict(np.array([s]), verbose=0)
        a = np.argmax(q_vals)
        self.prev_s = s
        self.prev_a = a

        return a

    def choose_action(self, s):
        if random.random() < self.epsilon:
            result =  np.random.randint(self.action_dim)
            print("Returning", result)
            return result
        else:
            q_vals = self.model.predict(np.array([s]), verbose=0)
            result = np.argmax(q_vals)
            print("Returning", result)
            return result