'''
Algorithm for Deep Q-Learning :
1. Initialize replay memory capacity.
2. Initialize the network with random weights.
3. For each episode:
    1. Initialize the starting state.
    2. For each time step:
        1. Select an action.
            ->Via exploration or exploitation
        2. Execute selected action in an emulator.
        3. Observe reward and next state.
        4. Store experience in replay memory.
        5. Sample random batch from replay memory.
        6. Preprocess states from batch.
        7. Pass batch of preprocessed states to policy network.
        8. Calculate loss between output Q-values and target Q-values.
            ->Requires a second pass to the network for the next state
        9. Gradient descent updates weights in the policy network to minimize loss.
'''

import tensorflow as tf
import gym
import numpy as np
from tqdm import tqdm
from collections import namedtuple, deque
import math
import random
from itertools import count

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 100000)
        self.discount_rate = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.00
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.001
        self.policy_model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.batch_size = 256

    def update_target_model(self):
        self.target_model.set_weights(self.policy_model.get_weights())

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32, input_dim= self.state_size, kernel_initializer= 'he_uniform', activation= 'relu'))
        model.add(tf.keras.layers.Dense(24, kernel_initializer= 'he_uniform', activation= 'relu'))
        model.add(tf.keras.layers.Dense(self.action_size, kernel_initializer= 'he_uniform', activation= 'relu'))
        model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= self.learning_rate), loss= 'mse')
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done, current_episode):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_end:
            self.epsilon = self.epsilon_end + (self.epsilon_end + self.epsilon) * math.exp(-1 * current_episode * self.epsilon_decay)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.action_size))
        else:
            act_values = self.policy_model.predict(state)[0]
            return np.argmax(act_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done =[], [], []
        for i in range(batch_size):
            update_input[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            update_target[i] = minibatch[i][3]
            done.append(minibatch[i][4])
        target = self.policy_model.predict(update_input)
        target_val = self.target_model.predict(update_target)
        for i in range(batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + (self.discount_rate * np.amax(target_val[i]))
        self.policy_model.fit(update_input, target, epochs=1, verbose=0)


if __name__ == '__main__':
    NUM_EPISODES = 500
    ENV = gym.make('CartPole-v0')
    STATE_SIZE = ENV.observation_space.shape[0]
    ACTION_SIZE = ENV.action_space.n
    agent = DQN(STATE_SIZE, ACTION_SIZE)
    scores = []
    BATCH_SIZE = 256
    done = False
    for e in tqdm(range(NUM_EPISODES)):
        state = ENV.reset()
        state = np.reshape(state, [1, STATE_SIZE])
        score = 0
        render_start = False
        render_stop = False
        for time_p in count():
            if render_start:
                ENV.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = ENV.step(action)
            next_state = np.reshape(next_state, [1, STATE_SIZE])
            reward = reward if not done or score == 499 else -100
            agent.remember(state, action, reward, next_state, done, e)
            agent.replay()
            score += reward
            state = next_state

            if done:
                agent.update_target_model()
                score = score if score == 500 else score+100
                scores.append(score)
                #print("Episode: {}/{}, Score: {}, Epsilon: {:.2}".format(e, NUM_EPISODES, score, agent.epsilon))
                break

        if render_stop:
            ENV.close()

    print(scores)