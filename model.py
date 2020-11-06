import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Reshape(nn.Module):
    """Simply reshapes input into desired shape"""
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class MusicAutoEncoder(pl.LightningModule):
# uses pytorch_lightning -- https://pytorch-lightning.readthedocs.io/en/stable/new-project.html
    def __init__(self, n_features=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,10), stride=(1,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=(3,10)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=(1,5)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, kernel_size=(1,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            # Reshape(-1, 7, 36),
            nn.Conv2d(128, 128, (7, 36)), # basically pools them into 128x1x1
            nn.ReLU(),
            # Reshape(-1, 128)
        )

        self.echonest_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
            nn.Sigmoid() # into [0,1]
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # train steps are in training_step
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        # x: torch.Size([N, n_mels, 1271]) y: torch.Size([N, 20])
        x = x.unsqueeze(1) # add fake channel dimension
        z = self.encoder(x)
        z = z.squeeze()
        feature_prediction = self.echonest_predictor(z)
        feature_loss = F.mse_loss(feature_prediction, y)

        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        loss = feature_loss
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.unsqueeze(1) # add fake channel dimension
        z = self.encoder(x)
        z = z.squeeze()
        feature_prediction = self.echonest_predictor(z)
        return feature_prediction # TODO: return autoencode prediction here too

    # def validation_epoch_end(self, val_step_outputs):
    #     for pred in val_step_outputs:
    #         # do something with all the predictions from each validation_step

    # def test_step(self, batch, batch_idx):

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer