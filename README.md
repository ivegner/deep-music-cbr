# deep-music-cbr
Deep content-based music recommendation using a two-headed CNN architecture. Project for UMass Amherst CS682.

## TODO:
- Model saving (in lightning_logs already?)
- Latent space operations
- Autoencoder head
- Early stopping

## Experiment log
All runs in lightning_logs, can be accessed with tensorboard by `tensorboard --logdir ./lightning_logs`

Run 1: Baseline. 46% accuracy.
Run 2: As above, but with batchnorm. Val loss doesn't decrease, accuracy steady at 25%.