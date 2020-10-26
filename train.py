from dataset import MusicDataModule
from model import MusicAutoEncoder
import pytorch_lightning as pl

pl.seed_everything(69)

BATCH_SIZE = 32

dataset = MusicDataModule(batch_size=batch_size)
model = MusicAutoEncoder()
trainer = pl.Trainer()
trainer.fit(model, dataset)
# tensorboard with `tensorboard --logdir ./lightning_logs`