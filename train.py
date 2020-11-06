import pytorch_lightning as pl

from dataset import MusicDataModule
from model import MusicAutoEncoder

pl.seed_everything(69)

BATCH_SIZE = 32

dataset = MusicDataModule(batch_size=BATCH_SIZE, num_workers=12)
dataset.prepare_data() # temporary for debugging of dataset
model = MusicAutoEncoder() # TODO: get n_features from dataset
trainer = pl.Trainer(gpus=1)
trainer.fit(model, dataset)
# tensorboard with `tensorboard --logdir ./lightning_logs`
