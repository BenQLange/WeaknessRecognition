import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as math
import math
import random

from DRQN import DRQN

class Agent(object):
    def __init__(self, n_actions, max_num_episodes=50, max_episode_length=300):
        self.n_actions = n_actions
        self.max_num_episodes = max_num_episodes
        self.max_episode_length = max_episode_length
        self.network = DRQN(self.n_actions)
        self.gamma = 0.9
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self.visited = []

    def train(self, observation, action, reward, new_observation, hidden):
        obs_list = []
        action_list = []
        reward_list = []

        obs_list.append(observation)
        obs_list.append(new_observation)
        action_list.append(action)
        reward_list.append(reward)

        Q, _ = self.network.forward(observation, hidden)
        Q_next, _ = self.network.forward(new_observation, hidden)
        Q_est = Q_next.clone()
        print("Q_learning")
        print(Q.shape)
        max_next_q = torch.max(Q_est[0, 0, :]).clone().detach()
        Q_est[0, 0, action] = reward + self.gamma * max_next_q

        loss = self.loss(Q, Q_est)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()


    def get_action(self, obs, hidden, epsilon):

        if random.random() > epsilon:
            print("HEREREER")
            q, new_hidden = self.network.forward(obs, hidden)
            top_actions = q[0].topk(10)[1].data
            print(list(top_actions[0].numpy()))
            for action in list(top_actions[0].numpy()):
                if action not in self.visited:
                    self.visited.append(action)
                    return action, new_hidden

        q, new_hidden = self.network.forward(obs, hidden)
        action = random.randint(0, self.n_actions-1)

        self.visited.append(action)

        return action, new_hidden
