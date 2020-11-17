# deep-music-cbr
Deep content-based music recommendation using a two-headed CNN architecture. Project for UMass Amherst CS682.

## TODO:
- Model saving (in lightning_logs already?)
- Latent space operations
- Autoencoder head
- Early stopping

## Experiment log
All runs in lightning_logs, can be accessed with tensorboard by `tensorboard --logdir ./lightning_logs`

Run 43: First successful training run. Trained on FMA small, genre only. Learning rate 1e-4, best validation genre accuracy around 41%. 
Run 49: Swapped for Conv1D, no further changes. Faster convergence, better accuracy (42%).