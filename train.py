import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser

from dataset import MusicDataModule
from model import MusicAutoEncoder

pl.seed_everything(69)

parser = ArgumentParser()
parser.add_argument("--autoencode", action="store_true", default=False)
parser.add_argument("--dropout-enc", type=float, default=0.0)
parser.add_argument("--dropout-pred", type=float, default=0.0)
parser.add_argument("--load-from", type=str)
parser.add_argument("--resume-from", type=str, default=None)
parser.add_argument("-b", "--batch-size", type=int, default=16)
parser.add_argument("--n-subset", type=int, default=None)
parser.add_argument("--max-epochs", type=int, default=1000)
parser.add_argument("--no-stop", action="store_true", default=False)
parser.add_argument("--rebuild-existing", action="store_true", default=False)
args = parser.parse_args()

USE_SMALL = True
dataset = MusicDataModule(
    fma_small=USE_SMALL,
    batch_size=args.batch_size,
    num_workers=12,
    rebuild_existing=args.rebuild_existing,
)
n_genres = dataset.prepare_data(n_subset=args.n_subset)
if args.load_from:
    model = MusicAutoEncoder.load_from_checkpoint(args.load_from, n_genres=n_genres)
else:
    model = MusicAutoEncoder(
        n_genres=n_genres,
        do_autoencode=args.autoencode,
        encoder_dropout=args.dropout_enc,
        predictor_dropout=args.dropout_pred,
    )

trainer = pl.Trainer(
    gpus=1,
    callbacks=None if args.no_stop else [EarlyStopping(monitor="val_feature_loss", patience=5)],
    max_epochs=args.max_epochs,
    resume_from_checkpoint=args.resume_from
)
trainer.fit(model, dataset)
# tensorboard with `tensorboard --logdir ./lightning_logs`
