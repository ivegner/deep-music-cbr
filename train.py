import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser

from dataset import MusicDataModule
from model import MusicAutoEncoder

pl.seed_everything(69)

parser = ArgumentParser()
parser.add_argument("--load-from", type=str)
args = parser.parse_args()

BATCH_SIZE = 32
USE_ECHONEST = False
dataset = MusicDataModule(
    use_echonest=USE_ECHONEST, batch_size=BATCH_SIZE, num_workers=12, rebuild_existing=False
)
n_features, n_genres = dataset.prepare_data()  # temporary for debugging of dataset
if args.load_from:
    model = MusicAutoEncoder.load_from_checkpoint(
        args.load_from, n_features=n_features, n_genres=n_genres, use_echonest=USE_ECHONEST
    )
else:
    model = MusicAutoEncoder(n_features=n_features, n_genres=n_genres, use_echonest=USE_ECHONEST)
trainer = pl.Trainer(gpus=1, callbacks=[EarlyStopping(monitor="val_feature_loss", patience=7)])
trainer.fit(model, dataset)
# tensorboard with `tensorboard --logdir ./lightning_logs`
