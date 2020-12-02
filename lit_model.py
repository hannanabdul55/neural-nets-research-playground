import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


class LitModel(pl.LightningModule):

    def __init__(self, model, criterion=nn.CrossEntropyLoss(),name='def'):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.n = name
        self.accuracy = pl.metrics.Accuracy()
        self.train_accuracy = pl.metrics.Accuracy()

    def training_step(self, batch, batch_size):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log(self.n + ':train_loss', loss)
        self.train_accuracy(logits, y)
        self.log(self.n + ':train_acc', self.train_accuracy, on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.accuracy(logits, y)
        self.log(self.n + ':val_loss', loss)
        self.log(self.n + ':val_acc', self.accuracy, on_epoch=False, on_step=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     loss = self.criterion(self.model(x), y)
    #     self.log(self.n + ':test_loss', loss)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
