import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
import gzip
import pickle
import os
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score


def print_stats(model_p, sparsity=1e-4):
    params = torch.cat([param.view(-1) for param in list(model_p.parameters())])
    params_n = params.detach().numpy()
    N = params_n.size
    stats = {}
    stats['mean'] = np.mean(np.abs(params_n))
    print("Mean: ",stats['mean'])
    stats['std'] = np.sqrt(np.var(np.abs(params_n)))
    print("Standard Deviation: ",  stats['std'])
    print("Sparsity: ", np.sum(np.abs(params_n)<sparsity)/N)
    return stats

    
def get_model_error(model_t, testloader):
    tot_num = 0.0
    num_wrong = 0.0
    dataiter = iter(testloader)
    for images,labels in dataiter:
        outputs = model_t(images)
        for image, label, i in zip(images, labels, range(len(images))):
    #         imshow(image)
    #         print(model(image).detach().numpy())
    #         expected_class = classes[outputs[i].detach().numpy().argmax()]
            expected_class_i = int(outputs[i].detach().numpy().argmax())
    #         print("Predicted class: %s" % expected_class)
    #         print("expected: " + str(int(label)))
            tot_num+=1
            if int(label) != expected_class_i:
                num_wrong+=1
    error = num_wrong/tot_num
    print("test error: " + str(error))
    return error

def get_predictions_ys(model_t, dataiter):
    y, pred_y = [], []
    for images,labels in dataiter:
        outputs = model_t(images)
        for image, label, i in zip(images, labels, range(len(images))):
    #         imshow(image)
    #         print(model(image).detach().numpy())
    #         expected_class = classes[outputs[i].detach().numpy().argmax()]
            expected_class_i = int(outputs[i].detach().numpy().argmax())
            y.append(label.numpy())
            pred_y.append(expected_class_i)
    return y, pred_y


def train_model(model, traindata, criterion, optimizer=None, epochs=5):
    writer = SummaryWriter(flush_secs=10)
    loss_history = []
    
    if optimizer is None:
        optimizer = optim.Adam(list(model.parameters()), lr=3e-4)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(traindata, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if writer is not None:
                writer.add_scalar("Loss/Train", loss.item())

            # print statistics
    #         running_loss += loss.item()
            if i%10 == 9:
                loss_history.append(loss.item())
            if i % 300 ==299:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss.item()))
    return loss_history

def train_model_self_reg(model, traindata, criterion, optimizer=None, epochs=5):
    mse_loss = torch.nn.MSELoss(reduction='sum')
    writer = SummaryWriter(flush_secs=10)
    loss_history = []
    if optimizer is None:
        optimizer = optim.Adam(list(model.parameters()), lr=3e-4)
    model_param_prev = None
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(traindata, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels, model)
            if model_param_prev is not None:
                loss += mse_loss(get_vector_from_params(model),model_param_prev)
            loss.backward()
            model_param_prev = torch.cat([param.view(-1) for param in model.parameters()])
            optimizer.step()
            if writer is not None:
                writer.add_scalar("Loss/Train", loss.item())

            # print statistics
    #         running_loss += loss.item()
            if i%10 == 9:
                loss_history.append(loss.item())
            if i % 300 ==299:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss.item()))
    return loss_history

def get_vector_from_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters())

def set_params(params, model):
    return torch.nn.utils.vectors_to_parameters(params, model.parameters())

def get_params_size(model):
    return np.sum([len(param.view(-1)) for param in model.parameters()])
    
def get_encoder_model(model):
    N= get_params_size(model)
    return nn.Sequential(nn.Linear(N, int(N/2)),
                                      nn.ReLU(),
#                                       nn.Linear(int(N/1.2), int(N/1.3)),
#                                       nn.ReLU(),
#                                       nn.Linear(int(N/1.3), int(N/1.2)),
#                                       nn.ReLU(),
                                      nn.Linear(int(N/2), N)
                        )

def load_CIFAR(batch_size=64):
    transform = transforms.Compose(
    [transforms.ToTensor()#,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # print(torchvision.datasets.__dict__)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

def classification_model_accuracy(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None), accuracy_score(y_true, y_pred)

def plot_graph(x_values, y_values, x_label='', y_label='', fig_title='',plt_title='', color='g', show_legend=True):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1)
    axs.plot(x_values, y_values,color=color)
    axs.set(xlabel=x_label, ylabel=y_label, title=plt_title)
    if show_legend:
        axs.legend()
    fig.suptitle(fig_title)
    plt.show()
