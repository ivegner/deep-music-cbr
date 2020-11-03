import pytorch_lightning as pl

from dataset import MusicDataModule
from model import MusicAutoEncoder

pl.seed_everything(69)

BATCH_SIZE = 32

dataset = MusicDataModule(batch_size=BATCH_SIZE)
dataset.prepare_data(n_subset=10) # temporary for debugging of dataset
model = MusicAutoEncoder()
trainer = pl.Trainer()
trainer.fit(model, dataset)
# tensorboard with `tensorboard --logdir ./lightning_logs`
