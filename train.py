import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataset import MusicDataModule
from model import MusicAutoEncoder

pl.seed_everything(69)

BATCH_SIZE = 32
USE_ECHONEST = False
dataset = MusicDataModule(
    use_echonest=USE_ECHONEST, batch_size=BATCH_SIZE, num_workers=12, rebuild_existing=False
)
n_features, n_genres = dataset.prepare_data()  # temporary for debugging of dataset
model = MusicAutoEncoder(n_features=n_features, n_genres=n_genres, use_echonest=USE_ECHONEST)
trainer = pl.Trainer(gpus=1, callbacks=[EarlyStopping(monitor="val_feature_loss")])
trainer.fit(model, dataset)
# tensorboard with `tensorboard --logdir ./lightning_logs`
