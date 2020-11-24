import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torchvision

import os
import numpy as np

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 500

def train(epoch, model, optimizer, model_name, train_loader):
    train_losses = []
    train_counter = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.cuda())
        loss = F.nll_loss(output.cpu(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    torch.save(model.state_dict(), 'results/{}/{}_model.pth'.format(model_name, model_name))
    torch.save(optimizer.state_dict(), 'results/{}/{}_optimizer.pth'.format(model_name, model_name))
    np.save('results/{}/{}_params'.format(model_name, model_name), model.params)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_count = 0
    test_losses = []
    with torch.no_grad():
        for data, target in test_loader:
            total_count += 1
            output = model(data.cuda())
            test_loss += F.nll_loss(output.cuda(), target.cuda(), size_average=False).item()
            pred = output.cpu().data.max(1, keepdim=True)[1]
            correct += pred.eq(target.cpu().data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def test_sample(model, dataset, labels):
    model.eval()
    test_loss = 0
    correct = 0
    total_count = 0
    test_losses = []
    pred_history = []
    num_sample = len(dataset)
    with torch.no_grad():
        for data, target in zip(dataset,labels):
            data = torch.from_numpy(data)
            target = torch.from_numpy(target)
            total_count += 1
            output = model(data.cuda())
            loss_comp = F.nll_loss(output.cuda(), target.cuda(), size_average=False).item()
            test_losses.append(loss_comp)
            test_loss += loss_comp
            pred = output.cpu().data.max(1, keepdim=True)[1]
            pred = pred.eq(target.cpu().data.view_as(pred)).sum()
            pred_history.append(pred.item())
            correct += pred

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(dataset),
    100. * correct / len(dataset)))

    return test_losses, pred_history

def save_sample_test_set(indices):
    data_list = []
    target_list = []
    idx = 0
    with torch.no_grad():
        for data, target in test_loader:
            if idx in indices:
                data_list.append(data.numpy())
                target_list.append(target.numpy())
            idx += 1

    return (data_list,target_list)

def load_model(model_name):
    params = np.load('results/{}/{}_params.npy'.format(model_name, model_name))
    new_model = Net(params).cuda()
    network_state_dict = torch.load('results/{}/{}_model.pth'.format(model_name, model_name))
    new_model.load_state_dict(network_state_dict)
    return new_model

def setup_testset_sample():
    data_idx = list(np.random.randint(0, 10000, 50, dtype=int))
    data, labels = save_sample_test_set(data_idx)
    np.save('sample_test_set_data', data)
    np.save('sample_test_set_labels', labels)

class Net(nn.Module):
    """
    Basic Classifier
    """
    def __init__(self, params):
        super(Net, self).__init__()
        input_size = 784
        self.params = params
        output_size = 10

        self.model = nn.Sequential(nn.Linear(input_size, self.params[0]),
                      nn.ReLU(),
                      nn.Linear(self.params[0], self.params[1]),
                      nn.ReLU(),
                      nn.Linear(self.params[1], output_size),
                      nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)
