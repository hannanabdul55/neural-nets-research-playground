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
        self.activation = nn.Tanh()
        self.writer = writer
        self.register_buffer('alpha', torch.tensor(alpha, requires_grad=False))
        
    def forward(self, input):
        res = self.activation(self.alpha * input)
        if self.training:
            self.alpha = (self.momentum * self.alpha + (1-self.momentum)* (torch.std(input))).detach()
            if self.writer is not None:
                self.writer.add_scalar("Loss/Alpha",self.alpha.numpy())
#             print("alpha:" + str(self.alpha))
            
        return res
    
#     def forward(self, input):
        
       