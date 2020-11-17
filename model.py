import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import accuracy


class MusicAutoEncoder(pl.LightningModule):
    # uses pytorch_lightning -- https://pytorch-lightning.readthedocs.io/en/stable/new-project.html
    def __init__(self, n_genres, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 256, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(256, 256, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(256, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 32, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
        )

        self.feature_predictor = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_genres),
            nn.Sigmoid(),  # into [0,1]
        )
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        self.n_genres = n_genres
        self.learning_rate = learning_rate

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # train steps are in training_step
        x = x.unsqueeze(1)  # add fake channel dimension
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
        # print(z.shape)
        z = z.squeeze()
        genre_predictions = self.feature_predictor(z)
        val_feature_loss = F.cross_entropy(genre_predictions, y)

        assert genre_predictions.shape[1] == self.n_genres
        genre_predictions = torch.argmax(genre_predictions, dim=1)

        self.log("val_feature_loss", val_feature_loss)

        return (genre_predictions.detach().cpu(), y.detach().cpu())  # TODO : return autoencode prediction here too

    def validation_epoch_end(self, val_step_outputs):
        genre_preds, genre_ys = zip(*val_step_outputs)
        genre_preds, genre_ys = torch.cat(genre_preds), torch.cat(genre_ys)
        genre_accuracy = accuracy(
            genre_preds, genre_ys, num_classes=self.n_genres, class_reduction="weighted"
        )

        self.log("val_genre_accuracy", genre_accuracy)
        print("Val genre accuracy:", genre_accuracy[0].cpu().numpy())

    # def test_step(self, batch, batch_idx):

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
