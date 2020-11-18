# deep-music-cbr
Deep content-based music recommendation using a two-headed CNN architecture. Project for UMass Amherst CS682.

## TODO:
- Model saving (in lightning_logs already?)
- Latent space operations
- Autoencoder head
- Early stopping

## Experiment log
All runs in lightning_logs, can be accessed with tensorboard by `tensorboard --logdir ./lightning_logs`

"Baseline" run: Baseline. 46% accuracy.

"Batchnorm" run: As above, but with batchnorm. Val loss doesn't decrease, accuracy steady at 25%.

"Dropout": Baseline with 25% dropout. Comparable performance to baseline, but achieves 51% after 100 epochs of training.

