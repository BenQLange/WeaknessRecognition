import  numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import MNISTClassifier
import envs
import agents_maml
import copy

from DRQN import *

def get_decay(epi_iter):
    decay = math.pow(0.99999999, epi_iter)
    if decay < 0.01:
        decay = 0.01
    return decay

class Meta(nn.Module):
    """
    Adapted MAML for RL.
    Inspired by https://github.com/dragen1860/MAML-Pytorch
    """
    def __init__(self, models, test_model, update_lr=0.01, meta_lr=0.001, update_step=50,
        update_step_test=50, dataset_path="sample_test_set"):

        super(Meta, self).__init__()

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.task_num = len(models)
        self.update_step = update_step
        self.update_step_test = update_step_test
        self.n_actions = 50
        self.envs = []

        for model in models:
            self.envs.append(envs.SUTEnv(model))

        self.env_test = envs.SUTEnv(test_model)
        self.agent = agents_maml.AgentMAML(self.n_actions)

        self.net = DRQN(self.n_actions) #MAKE SURE YOU CAN SET THE WEIGHTS
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)

        #load model explicitly or just env>
        self.dataset = np.load(dataset_path+"_data.npy") #[Num Samples, 1, 1, 28, 28]
        self.dataset_labels = np.load(dataset_path+"_labels.npy")

    def forward(self):

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]



        for env in self.envs: #Equivalent of num tasks

            hidden = (Variable(torch.zeros(1, 1, 50).float()), Variable(torch.zeros(1, 1, 50).float()))

            # 1. run the i-th task and compute loss for k=0
            observation = env.get_obs()

            self.agent.network = self.net
            action, hidden = self.agent.get_action(observation, hidden, get_decay(0)) #CHANGE STEP
            reward, succ = env.step(action)

            new_observation = env.get_obs()

            #Update the network with correct params
            loss = self.agent.loss(observation, action, reward, new_observation, hidden)
            grad = torch.autograd.grad(loss, self.net.parameters(), allow_unused=True, retain_graph=True)

            fast_weights = list(map(lambda p: torch.nn.Parameter(p[1] - self.update_lr * p[0]), zip(grad, self.net.parameters())))

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                observation = env.get_obs()
                action, hidden = self.agent.get_action(observation, hidden, get_decay(k), fast_weights) #CHANGE STEP
                reward, succ = env.step(action)

                new_observation = env.get_obs()

                #Update the network with correct params
                loss = self.agent.loss(observation, action, reward, new_observation, hidden, fast_weights)

                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: torch.nn.Parameter(p[1] - self.update_lr * p[0]), zip(grad, self.net.parameters())))

                if k == self.update_step - 1:
                    observation = env.get_obs()
                    action, hidden = self.agent.get_action(observation, hidden, get_decay(k), fast_weights) #CHANGE STEP
                    reward, succ = env.step(action)

                    new_observation = env.get_obs()

                    #Update the network with correct params
                    loss_q = self.agent.loss(observation, action, reward, new_observation, hidden, fast_weights)
                    losses_q[k+1] += loss_q


        # end of all tasks
        loss_q = losses_q[-1] / self.task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward(retain_graph=True)
        self.meta_optim.step()

        print("Training finished")

    def finetunning(self):

        ### MODDDIFFFY NETWORK

        hidden = (Variable(torch.zeros(1, 1, 50).float()), Variable(torch.zeros(1, 1, 50).float()))

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = copy.deepcopy(self.net)
        self.agent.network = net

        observation = self.env_test.get_obs()
        action, hidden = self.agent.get_action(observation, hidden, get_decay(0)) #CHANGE STEP
        reward, succ = self.env_test.step(action)
        new_observation = self.env_test.get_obs()

        #Update the network with correct params
        loss = self.agent.loss(observation, action, reward, new_observation, hidden)
        grad = torch.autograd.grad(loss, net.parameters(), allow_unused=True, retain_graph=True)
        fast_weights = list(map(lambda p: torch.nn.Parameter(p[1] - self.update_lr * p[0]), zip(grad, self.net.parameters())))

        succ_rate = []
        num_succ = 0

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            observation = self.env_test.get_obs()
            action, hidden = self.agent.get_action(observation, hidden, get_decay(k), fast_weights) #CHANGE STEP
            reward, succ = self.env_test.step(action)
            new_observation = self.env_test.get_obs()

            if succ == True:
                num_succ += 1

            succ_rate.append(num_succ)

            #Update the network with correct params
            loss = self.agent.loss(observation, action, reward, new_observation, hidden, fast_weights)

            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: torch.nn.Parameter(p[1] - self.update_lr * p[0]), zip(grad, self.net.parameters())))


        return succ_rate, self.env_test
