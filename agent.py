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

    def train(self):
        obs_list = []
        action_list = []
        reward_list = []

        buffer = self.buffer.sample()
        for i in range(len(buffer)):
            obs_list.append(buffer[i][0])
            action_list.append(buffer[i][1])
            reward_list.append(buffer[i][2])

        obs_list = self.img_list_to_batch(obs_list)
        hidden = (Variable(torch.zeros(1, 1, 16).float()), Variable(torch.zeros(1, 1, 16).float()))
        Q, hidden = self.drqn.forward(obs_list, hidden)
        Q_est = Q.clone()
        for t in range(len(memo) - 1):
            max_next_q = torch.max(Q_est[t+1, 0, :]).clone().detach()
            Q_est[t, 0, action_list[t]] = reward_list[t] + self.gamma * max_next_q
        T = len(memo) - 1
        Q_est[T, 0, action_list[T]] = reward_list[T]

        loss = self.loss_fn(Q, Q_est)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_action(self, obs, hidden, itr):
        epsilon = get_decay(itr)
        if random.random() > epsilon:
            q, new_hidden = self.network.forward(obs, hidden)
            action = q[0].max(1)[1].data[0].item()
        else:
            q, new_hidden = self.network.forward(obs, hidden)
            action = random.randint(0, self.n_actions-1)
        return action, new_hidden

def get_decay(epi_iter):
    decay = math.pow(0.999, epi_iter)
    if decay < 0.05:
        decay = 0.05
    return decay

if __name__ == '__main__':
    random.seed()
    env = EnvTMaze(4, random.randint(0, 1))
    max_epi_iter = 30000
    max_MC_iter = 100
    agent = Agent(N_action=4, max_epi_num=5000, max_epi_len=max_MC_iter)
    train_curve = []

    for epi_iter in range(max_epi_iter):
        random.seed()
        env.reset(random.randint(0, 1))
        hidden = (Variable(torch.zeros(1, 1, 16).float()), Variable(torch.zeros(1, 1, 16).float()))
        for MC_iter in range(max_MC_iter):
            obs = env.get_obs()
            action, hidden = agent.get_action(obs, hidden, get_decay(epi_iter))
            reward = env.step(action)
            agent.remember(obs, action, reward)
            if reward != 0 or MC_iter == max_MC_iter-1:
                agent.buffer.create_new_epi()
                break
        print('Episode', epi_iter, 'reward', reward, 'where', env.if_up)
        if epi_iter % 100 == 0:
            train_curve.append(reward)
        if agent.buffer.is_available():
            agent.train()
    np.save("drqn_test.npy", np.array(train_curve))
