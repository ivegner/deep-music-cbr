import pytorch_lightning as pl

from argparse import ArgumentParser

from dataset import MusicDataModule
from models.autoencoder import MusicAutoEncoder

pl.seed_everything(69)

parser = ArgumentParser()
parser.add_argument("load_filename", type=str, required=True, help="File path to the model checkpoint to load")
parser.add_argument("songs", nargs="+", type=str, required=True, help="One or more songs to use as sources for analysis")
args = parser.parse_args()

model = MusicAutoEncoder.load_from_checkpoint(
    args.load_filename, n_features=n_features, n_genres=n_genres
)
