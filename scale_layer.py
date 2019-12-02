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


class ScaleLayer(nn.Module):
    def __init__(self, momentum=0.95, alpha=1.0, writer=None):
        super().__init__()
        self.momentum = momentum
        self.activation = nn.ReLU()
        self.writer = writer
#         self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('alpha', torch.tensor(alpha, requires_grad=True))
        
    def forward(self, input):
        res = self.activation(input/self.alpha)
        if self.training:
            self.alpha = nn.Parameter((self.momentum * self.alpha + (1-self.momentum)* (torch.std(input))))
        if self.writer is not None:
            self.writer.add_scalar("Loss/Alpha",self.alpha.data)
#             print("alpha:" + str(self.alpha))
        return res
    
#     def forward(self, input):
        
       