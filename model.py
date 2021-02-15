import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import accuracy
from torch.nn.init import kaiming_uniform_


class MusicAutoEncoder(pl.LightningModule):
    # uses pytorch_lightning -- https://pytorch-lightning.readthedocs.io/en/stable/new-project.html
    def __init__(
        self,
        n_genres,
        do_autoencode=False,
        learning_rate=1e-4,
        encoder_dropout=0,
        predictor_dropout=0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.ModuleList(
            [
                ConvBlock(
                    1,
                    512,
                    conv_kernel_shape=(3, 11),
                    conv_stride_shape=(2, 5),
                    pool_kernel_shape=(1, 1),
                    conv_padding_shape=(1, 5),
                    dropout=encoder_dropout,
                ),
                ConvBlock(
                    512,
                    256,
                    conv_kernel_shape=(3, 3),
                    pool_kernel_shape=(2, 2),
                    dropout=encoder_dropout,
                    conv_padding_shape=(1, 1),
                ),
                ConvBlock(
                    256,
                    256,
                    conv_kernel_shape=(3, 3),
                    pool_kernel_shape=(2, 2),
                    dropout=encoder_dropout,
                    conv_padding_shape=(1, 1),
                ),
                ConvBlock(
                    256,
                    128,
                    conv_kernel_shape=(3, 3),
                    pool_kernel_shape=(2, 2),
                    dropout=encoder_dropout,
                    conv_padding_shape=(1, 1),
                ),
                ConvBlock(
                    128,
                    64,
                    conv_kernel_shape=(3, 3),
                    pool_kernel_shape=(2, 2),
                    dropout=encoder_dropout,
                    conv_padding_shape=(1, 1),
                ),
                ConvBlock(
                    64,
                    32,
                    conv_kernel_shape=(3, 3),
                    pool_kernel_shape=(2, 2),
                    conv_padding_shape=(1, 1),
                    dropout=encoder_dropout,
                ),
                ConvBlock(
                    32,
                    32,
                    conv_kernel_shape=(2, 4),
                    pool_kernel_shape=(1, 5),
                    conv_padding_shape=(0, 0),
                    dropout=0,
                ),
            ]
        )
        self.feature_predictor = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(64, n_genres),
            nn.Sigmoid(),  # into [0,1]
        )
        if do_autoencode:
            self.decoder = build_decoder(self.encoder)

        self.n_genres = n_genres
        self.learning_rate = learning_rate
        self.do_autoencode = do_autoencode

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # train steps are in training_step
        z, _ = self.encode(x)
        return z

    def encode(self, x):
        """
        Run the encoder pass on x
        """
        x = x.unsqueeze(1)  # add fake channel dimension
        z = x
        # print("Encoding start", z.shape)
        for block in self.encoder:
            z = block(z)
            # print("Encoding out", z.shape)
        return z

    def decode(self, z):
        """
        Run the decoder pass on z
        """
        k = z
        for i, block in enumerate(self.decoder):
            k = block(k)
        return k

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        # x: torch.Size([N, n_mels, 1271]) y: torch.Size([N, n_features])
        z = self.encode(x)

        feature_prediction = self.feature_predictor(z.squeeze())
        feature_loss = F.cross_entropy(feature_prediction, y)
        self.log("feature_loss", feature_loss)
        loss = feature_loss

        if self.do_autoencode:
            x_hat = self.decode(z).squeeze(1) # remove channels
            autoencoder_loss = F.mse_loss(x_hat, x)
            self.log("autoencoder_loss", autoencoder_loss)
            loss += autoencoder_loss
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encode(x)

        genre_predictions = self.feature_predictor(z.squeeze())
        val_feature_loss = F.cross_entropy(genre_predictions, y)

        assert genre_predictions.shape[1] == self.n_genres
        genre_predictions = torch.argmax(genre_predictions, dim=1)

        self.log("val_feature_loss", val_feature_loss)

        return (
            genre_predictions.detach().cpu(),
            y.detach().cpu(),
        )  # TODO : return autoencode prediction here too

    def validation_epoch_end(self, val_step_outputs):
        genre_preds, genre_ys = zip(*val_step_outputs)
        genre_preds, genre_ys = torch.cat(genre_preds), torch.cat(genre_ys)
        genre_accuracy = accuracy(
            genre_preds, genre_ys, num_classes=self.n_genres, class_reduction="weighted"
        )

        self.log("val_genre_accuracy", genre_accuracy)
        # print("Val genre accuracy:", genre_accuracy.item())

    # def test_step(self, batch, batch_idx):

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class ConvBlock(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        conv_kernel_shape,
        pool_kernel_shape,
        conv_stride_shape=(1, 1),
        conv_padding_shape=(0, 0),
        dropout=0.0,
    ):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.conv_kernel_shape = conv_kernel_shape
        self.pool_kernel_shape = pool_kernel_shape
        self.conv_stride_shape = conv_stride_shape
        self.conv_padding_shape = conv_padding_shape
        self.dropout = dropout

        self.conv = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=conv_kernel_shape,
            stride=conv_stride_shape,
            padding=conv_padding_shape,
        )
        kaiming_uniform_(self.conv.weight)
        # nn.BatchNorm2d(64),
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool2d(pool_kernel_shape)

    def forward(self, x):
        c = self.conv(x)
        c = self.relu(c)
        c = self.dropout(c)
        c = self.pool(c)
        return c


class DeconvBlock(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        conv_kernel_shape,
        pool_kernel_shape,
        conv_stride_shape=(1, 1),
        conv_padding_shape=(0, 0),
        output_padding=(0, 0),
        dropout=0.0,
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            channels_in,
            channels_out,
            kernel_size=conv_kernel_shape,
            stride=conv_stride_shape,
            padding=conv_padding_shape,
            output_padding=output_padding,
        )
        kaiming_uniform_(self.deconv.weight)
        # nn.BatchNorm2d(64),
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.unpool = nn.Upsample(scale_factor=pool_kernel_shape)

    def forward(self, x):
        c = self.unpool(x)
        c = self.deconv(c)
        c = self.relu(c)
        c = self.dropout(c)
        return c


def build_decoder(encoder):
    """Builds decoder symmetrical to encoder.

    Encoder: nn.ModuleList
    """

    decoder = nn.ModuleList()
    for block in encoder[::-1]:
        block_padding = block.conv_padding_shape
        block_stride = block.conv_stride_shape
        block_kernel = block.conv_kernel_shape
        
        deconv_output_padding = (block_stride[0]-1, block_stride[1]-1)
        decoder.append(
            DeconvBlock(
                channels_in=block.channels_out,
                channels_out=block.channels_in,
                conv_kernel_shape=block.conv_kernel_shape,
                pool_kernel_shape=block.pool_kernel_shape,
                conv_stride_shape=block.conv_stride_shape,
                conv_padding_shape=block_padding,
                dropout=0,
                output_padding=deconv_output_padding,
            )
        )
    return decoder

