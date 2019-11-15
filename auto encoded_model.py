from utils import *

class AutoregularizedModel():
    def __init__(self, model):
        self.model = model
        N= len(model.parameters())
        self.regmodel = nn.Sequential(nn.Linear(N, int(N/1.2)),
                                      nn.ReLU(),
                                      nn.Linear(int(N/1.2), int(N/1.3)),
                                      nn.ReLU(),
                                      nn.Linear(int(N/1.3), int(N/1.2)),
                                      nn.ReLU(),
                                      nn.Linear(int(N/1.2), N)
                                     )
    
    def forward(self, X):
        y = self.model(X)