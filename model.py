import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import accuracy

class Reshape(nn.Module):
    """Simply reshapes input into desired shape"""

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class MusicAutoEncoder(pl.LightningModule):
    # uses pytorch_lightning -- https://pytorch-lightning.readthedocs.io/en/stable/new-project.html
    def __init__(self, n_features, n_genres, use_echonest=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 10), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 10)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 128, (7, 36)),  # basically pools them into 128x1x1
            nn.ReLU(),
        )

        self.feature_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
            nn.Sigmoid(),  # into [0,1]
        )
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        self.use_echonest = use_echonest
        self.n_genres = n_genres
        # n_genres used for calculating genre accuracy when there are more features than genre

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # train steps are in training_step
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        # x: torch.Size([N, n_mels, 1271]) y: torch.Size([N, n_features])
        x = x.unsqueeze(1)  # add fake channel dimension
        z = self.encoder(x)
        z = z.squeeze()
        feature_prediction = self.feature_predictor(z)
        if self.use_echonest:
            feature_loss = F.mse_loss(feature_prediction, y)
        else:
            y = torch.argmax(y, axis=1)
            feature_loss = F.cross_entropy(feature_prediction, y)


        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        loss = feature_loss
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.unsqueeze(1)  # add fake channel dimension
        z = self.encoder(x)
        z = z.squeeze()
        feature_prediction = self.feature_predictor(z)

        if self.use_echonest:
            genre_predictions = feature_prediction[:, -self.n_genres:]
            genre_y = y[:, -self.n_genres:]
        else:
            genre_predictions = feature_prediction
            genre_y = y
        assert genre_predictions.shape[1] == self.n_genres
        genre_predictions = torch.argmax(genre_predictions, dim=1)
        genre_y = torch.argmax(y, dim=1)
        return (genre_predictions, genre_y)  # TODO : return autoencode prediction here too

    def validation_epoch_end(self, val_step_outputs):
        genre_preds, genre_ys = zip(*val_step_outputs)
        genre_preds, genre_ys = torch.cat(genre_preds), torch.cat(genre_ys)
        genre_accuracy = accuracy(
            genre_preds, genre_ys, num_classes=self.n_genres, class_reduction="weighted"
        )
        self.log("val_genre_accuracy", genre_accuracy)

    # def test_step(self, batch, batch_idx):

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
