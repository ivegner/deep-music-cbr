import os
import warnings
from os.path import dirname, join

import torch
import librosa
import librosa.display
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import utils

# pylint:disable=attribute-defined-outside-init,invalid-name

load_dotenv(".env")

# Directory where mp3 are stored.
AUDIO_DIR = os.environ.get("AUDIO_DIR")
PREPPED_DATA_DIR = os.environ.get("PREPPED_DATA_DIR")
if not os.path.exists(PREPPED_DATA_DIR):
    os.makedirs(PREPPED_DATA_DIR)


class MusicDataModule(pl.LightningDataModule):
    def __init__(self, autoencode=False, batch_size=64, mfc_kwargs=None, num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.autoencode = autoencode
        self.mfc_kwargs = mfc_kwargs if mfc_kwargs is not None else {}
        self.mfc_kwargs.setdefault("n_fft", 2048)
        self.mfc_kwargs.setdefault("hop_length", 512)
        self.mfc_kwargs.setdefault("n_mels", 128)
        self.num_workers = num_workers

    def prepare_data(self, n_subset=None):
        # called only on 1 GPU
        # do things that are only done once here
        # if data isn't downloaded, download data
        # save data on disk
        tracks = utils.load("data/fma_metadata/tracks.csv")
        echonest = utils.load("data/fma_metadata/echonest.csv")
        if n_subset is not None:
            echonest = echonest.head(n_subset)

        ###### Find Echonest Tracks ######
        interesting_base_cols = ["genre_top"]  # , "genres"]
        # isolate tracks with echonest data
        # prune columns we don't want to train on (from the non-echonest data)
        tracks = tracks.loc[
            echonest.index, tracks.columns.get_level_values(1).isin(interesting_base_cols)
        ]
        # join with echonest information. Can also grab song hotttness if interested.
        tracks = tracks["track"].join(echonest[("echonest", "audio_features")], how="inner")

        ###### Clean and Prep Labels ######
        # clean songs with NA values (only genre can be missing).
        print("NA values per feature:", tracks.isna().sum(), sep="\n")
        tracks = tracks.dropna()
        print(f"Total clean EchoNest tracks: {len(tracks)}")
        print("Genre counts:", tracks.groupby(["genre_top"]).size(), sep="\n")
        # One-hot encode genres
        tracks = pd.get_dummies(tracks, columns=["genre_top"], prefix=["genre_is"])
        # scale tempo
        self.tempo_scaler = StandardScaler()
        tracks["tempo"] = self.tempo_scaler.fit_transform(tracks[["tempo"]])

        # incorporate track_id into columns and reset index to numeric
        tracks = tracks.reset_index(drop=False)

        ###### Build Features ######
        track_paths = []
        for i, track_info in tqdm(tracks.iterrows(), total=len(tracks)):
            track_id = int(track_info["track_id"])
            _, track_data_path = self._get_track_data_path(track_id)
            if os.path.exists(track_data_path):
                # has been built before, append and skip
                track_paths.append(track_data_path)
                continue
            # comment claimed that this function doesn't work correctly
            track_filename = utils.get_audio_path(AUDIO_DIR, track_id)
            # print(f"Processing {i}/{len(tracks)}, {track_filename=}", end="\r")
            with warnings.catch_warnings():
                # raises "UserWarning: PySoundFile failed. Trying audioread instead."
                # see https://github.com/librosa/librosa/issues/1015
                warnings.simplefilter("ignore")
                # load song audio and sample rate.
                # duration is fixed because some clips are just a little shorter

                try:
                    audio_data, sample_rate = librosa.load(
                        track_filename, sr=None, mono=True, duration=29.5
                    )
                except Exception as e:
                    # TODO: breaks on 025713
                    print(e)
                    continue
            track_x = self.build_track_features(audio_data, sample_rate)
            track_x = np.array(track_x, dtype=float)
            track_y = track_info.drop("track_id")
            # save data
            self.save_track_data(track_id, track_x, track_y)
            track_paths.append(track_data_path)
        self.track_paths = track_paths

    def _get_track_data_path(self, track_id):
        tid_str = "{:06d}".format(track_id)
        param_string = "_".join([f"{k}-{v}" for k, v in self.mfc_kwargs.items()])
        path = os.path.join(PREPPED_DATA_DIR, tid_str[:3], tid_str)
        return path, os.path.join(path, param_string + ".npz")

    def save_track_data(self, track_id, track_features, track_y):
        """
        Save track features to PREPPED_DATA_DIR, to be loaded later
        """
        data_dir, data_path = self._get_track_data_path(track_id)
        os.makedirs(data_dir, exist_ok=True)
        np.savez_compressed(data_path, X=track_features, y=track_y)

    def build_track_features(self, mp3, sample_rate):
        """
        Build numpy array of numerical features for a given track
        """
        # https://medium.com/@tanveer9812/mfccs-made-easy-7ef383006040
        # https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
        mfc = librosa.feature.melspectrogram(
            mp3,
            sr=sample_rate,
            n_fft=self.mfc_kwargs["n_fft"],
            hop_length=self.mfc_kwargs["hop_length"],
            n_mels=self.mfc_kwargs["n_mels"],
        )
        # TODO: normalization
        return mfc

    def setup(self, stage=None):
        # do splits, transforms, parameter-dependent stuff,
        # set relevant variables
        n_tracks = len(self.track_paths)
        train_frac, val_frac, test_frac = 0.8, 0.1, 0.1
        n_train, n_test = round(n_tracks * train_frac), round(n_tracks * test_frac)
        shuffled_paths = self.track_paths[:]
        np.random.shuffle(shuffled_paths)

        train_paths = shuffled_paths[:n_train]
        val_paths = shuffled_paths[n_train:-n_test]
        test_paths = shuffled_paths[-n_test:]
        if stage == 'fit' or stage is None:
            self.train = FMASplit(train_paths)
            self.val = FMASplit(val_paths)
        if stage == 'test' or stage is None:
            self.test = FMASplit(test_paths)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)


# guide on building a dataset: https://github.com/utkuozbulak/pytorch-custom-dataset-examples
class FMASplit(Dataset):
    """
    Music dataset sourced from FMA
    """

    def __init__(self, data_paths, autoencode=False):
        self.data_paths = data_paths
        self.autoencode = autoencode
        self.count = len(self.data_paths)

    def __getitem__(self, index):
        # get one item from data source by index
        # probably call ToTensor on the data
        loaded = np.load(self.data_paths[index])
        X, y = loaded["X"], loaded["y"]
        if self.autoencode:
            y = X
        return torch.tensor(X), torch.tensor(y)

    def __len__(self):
        return self.count

