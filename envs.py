import random
import numpy as np
import torch.nn.functional as F
import torch

import MNISTClassifier

class SUTEnv(object):
    def __init__(self, model_name, testset_path="sample_test_set", provide_image=False):

        ## Load the model
        self.model = MNISTClassifier.load_model(model_name).eval()

        # Freeze all models
        for param in self.model.parameters():
            param.requires_grad = False


        #Load the testset.
        self.testset = np.load(testset_path+"_data.npy") #[Num Samples, 1, 1, 28, 28]
        self.testset_labels = np.load(testset_path+"_labels.npy")
        self.n_actions = self.testset.shape[0]

        #Initialize observation
        ## Keep the count of success and failures
        ## Observation is our simulation log encoded as a one-hot vector.
        ## IDX: Scenario is not execut = 2, Sceario FAILED = 1, Sceario SUCCEDED = 0
        self.observation = np.zeros((self.n_actions,3))
        self.observation[:,2] = 1 # No scenarios were executed (2).
        self.action_log = []

        #Additional params.
        self.provide_image = provide_image

    def reset(self):
        # I don't need reset I think. At least for now.
        pass

    def step(self, action):

        reward = 0
        #Get a scenario from the dataset.
        sample = self.testset[action]

        with torch.no_grad():
            pred = self.model(torch.from_numpy(sample).cuda())

        pred.reshape((1,10)) #10 digits in MNIST
        ## Evaluate the scenario
        pred_label = pred.max(1, keepdim=True)[1]
        correct_label = self.testset_labels[action]
        print("Correct label: ", pred_label[0][0].cpu().numpy(), " ", correct_label[0])
        if correct_label[0] == pred_label[0][0].cpu().numpy():
            print("Succ")
            success = True
        else:
            success = False
        print('here')
        print(pred.shape)
        print(correct_label.shape)
        reward = F.nll_loss(pred.cuda(), torch.from_numpy(correct_label).cuda(), size_average=False).item()
        self.update_obs(action, success, reward)

        ## return a loss/reward metric
        return reward, success

    def update_obs(self, action, success, reward):
        self.observation[action,:] = np.zeros((3))

        if success == True:
            self.observation[action,0] = 1
        else: #Fail
            self.observation[action,1] = 1

        self.action_log.append((action,success, reward))

    def get_obs(self):
        if not self.provide_image:
            return self.observation
        else:
            pass
            # Probably return a tuple with image and count and stack later
