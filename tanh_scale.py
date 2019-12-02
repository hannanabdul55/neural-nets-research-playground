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


class HardTanh(nn.Module):
    def __init__(self, writer=None):
        super().__init__()
        self.momentum = 0.9
        self.activation = nn.ReLU()
        self.writer = writer
#         self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.register_buffer('alpha', torch.tensor(1.0, requires_grad=False))
        
    def forward(self, input):
        res = self.activation(input/self.alpha)
        if self.training:
            self.alpha = (self.momentum * self.alpha + (1-self.momentum)* (torch.std(input))).detach()
        if self.writer is not None:
            self.writer.add_scalar("Loss/Alpha",self.alpha.data)
            print("alpha:" + str(self.alpha))
#         res = torch.clamp(np.exp(-1)*input, -1, 1)
        return res
    
#     def forward(self, input):
        
       