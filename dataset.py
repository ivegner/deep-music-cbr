import os
import sys
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils
import traceback

# pylint:disable=attribute-defined-outside-init,invalid-name

load_dotenv(".env")

# Directory where mp3 are stored.
AUDIO_DIR = os.environ.get("AUDIO_DIR")
PREPPED_DATA_DIR = os.environ.get("PREPPED_DATA_DIR")
if not os.path.exists(PREPPED_DATA_DIR):
    os.makedirs(PREPPED_DATA_DIR)

# broken tracks:
# 25180
# 25176
# 25175
# 25174
# 25173
# With error:
# Track 25173 broke with error
# Traceback (most recent call last):
#   File "/home/elliot/anaconda3/lib/python3.8/site-packages/librosa/core/audio.py", line 146, in load
#     with sf.SoundFile(path) as sf_desc:
#   File "/home/elliot/anaconda3/lib/python3.8/site-packages/soundfile.py", line 629, in __init__
#     self._file = self._open(file, mode_int, closefd)
#   File "/home/elliot/anaconda3/lib/python3.8/site-packages/soundfile.py", line 1183, in _open
#     _error_check(_snd.sf_error(file_ptr),
#   File "/home/elliot/anaconda3/lib/python3.8/site-packages/soundfile.py", line 1357, in _error_check
#     raise RuntimeError(prefix + _ffi.string(err_str).decode('utf-8', 'replace'))
# RuntimeError: Error opening './data/fma_large/025/025173.mp3': File contains data in an unknown format.

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "/mnt/d/deep-music-cbr/dataset.py", line 62, in _build_track_features
#     audio_data, sample_rate = librosa.load(
#   File "/home/elliot/anaconda3/lib/python3.8/site-packages/librosa/core/audio.py", line 163, in load
#     y, sr_native = __audioread_load(path, offset, duration, dtype)
#   File "/home/elliot/anaconda3/lib/python3.8/site-packages/librosa/core/audio.py", line 187, in __audioread_load
#     with audioread.audio_open(path) as input_file:
#   File "/home/elliot/anaconda3/lib/python3.8/site-packages/audioread/__init__.py", line 116, in audio_open
#     raise NoBackendError()
# audioread.exceptions.NoBackendError

# 036102 breaks due to length


def _get_track_data_path(track_id, path_kwargs):
    tid_str = "{:06d}".format(track_id)
    param_string = "_".join([f"{k}-{v}" for k, v in path_kwargs.items()])
    path = os.path.join(PREPPED_DATA_DIR, tid_str[:3], tid_str)
    return os.path.join(path, param_string + ".npz")


def _build_features_for_file(track_filename, mfc_kwargs):
    with warnings.catch_warnings():
        # raises "UserWarning: PySoundFile failed. Trying audioread instead."
        # see https://github.com/librosa/librosa/issues/1015
        warnings.simplefilter("ignore")
        # load song audio and sample rate.
        # duration is fixed because some clips are just a little shorter
        DURATION_CLIP = 29.7 # magic number that gets the audio data to be 1280 in length
        audio_data, sample_rate = librosa.load(
            track_filename, sr=mfc_kwargs["resample_rate"], mono=True, duration=DURATION_CLIP,
        )
        if audio_data.shape[0] != DURATION_CLIP * mfc_kwargs["resample_rate"]:
            raise ValueError(
                f"Track {track_filename} has duration {audio_data.shape[0]/mfc_kwargs['resample_rate']}, not {DURATION_CLIP}. Rejecting.",
            )
    track_x = _build_mfc(audio_data, sample_rate, mfc_kwargs)
    track_x = np.array(track_x, dtype=float)
    return track_x


def _build_track_features(k):
    """
    Build numpy array of numerical features for a given track
    """
    # https://medium.com/@tanveer9812/mfccs-made-easy-7ef383006040
    # https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
    track_id, mfc_kwargs, rebuild_existing = k
    try:
        track_data_path = _get_track_data_path(track_id, mfc_kwargs)
        if os.path.exists(track_data_path):
            if rebuild_existing:
                # delete in case the data was bad, will be rebuilt anyway if good
                os.remove(track_data_path)
            else:
                # append and skip
                return (track_id, track_data_path)
        # comment claimed that this function doesn't work correctly
        track_filename = utils.get_audio_path(AUDIO_DIR, track_id)
        track_x = _build_features_for_file(track_filename, mfc_kwargs)
        # print(f"Processing {i}/{len(tracks)}, {track_filename=}", end="\r")
        # save data
        _save_track_data(track_data_path, track_x)
        # print(track_id, track_x.shape, track_info.shape, track_filename, sample_rate, flush=True)
        return (track_id, track_data_path)
    except Exception as e:
        print(f"Track {track_id} broke with error {e}", flush=True)
        return None


def _save_track_data(data_path, track_features):
    """
    Save track features to PREPPED_DATA_DIR, to be loaded later
    """
    data_dir = os.path.dirname(data_path)
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(data_path, X=track_features)


def _build_mfc(mp3, sample_rate, mfc_kwargs):
    mfc = librosa.feature.melspectrogram(
        mp3,
        sr=sample_rate,
        n_fft=mfc_kwargs["n_fft"],
        hop_length=mfc_kwargs["hop_length"],
        n_mels=mfc_kwargs["n_mels"],
    )
    # TODO: normalization
    return mfc


class MusicDataModule(pl.LightningDataModule):
    def __init__(
        self, fma_small=True, batch_size=64, rebuild_existing=False, mfc_kwargs=None, num_workers=1,
    ):
        super().__init__()
        self.fma_small = fma_small
        self.batch_size = batch_size
        self.rebuild_existing = rebuild_existing
        self.mfc_kwargs = mfc_kwargs if mfc_kwargs is not None else {}
        self.mfc_kwargs.setdefault("resample_rate", 22050)
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

        if self.fma_small:
            tracks = tracks[tracks["set", "subset"] <= "small"]
        if n_subset is not None:
            tracks = tracks.head(n_subset)

        # remove genres with no tracks (should be 8 genres for small)
        tracks[("track", "genre_top")] = tracks[
            ("track", "genre_top")
        ].cat.remove_unused_categories()
        n_genres = tracks[("track", "genre_top")].nunique()
        tracks = tracks[[("track", "genre_top")]]

        ###### Clean and Prep Labels ######
        # clean songs with NA values (only genre can be missing).
        print("NA values per feature:", tracks.isna().sum(), sep="\n")
        tracks = tracks.dropna()
        print(f"Total clean tracks: {len(tracks)}")
        print("Genre counts:", tracks.groupby([("track", "genre_top")]).size(), sep="\n")
        # One-hot encode genres
        # tracks = pd.get_dummies(tracks, columns=[("track", "genre_top")], prefix=["genre_is"])
        # categorically encode genres
        genres = tracks[("track", "genre_top")].cat.codes
        assert n_genres == np.max(genres) + 1
        print(f"Total {n_genres} genre features")

        # # incorporate track_id into columns and reset index to numeric
        # tracks = tracks.reset_index(drop=False)

        ###### Build Features ######
        processed_tracks = process_map(
            _build_track_features,
            ((t_id, self.mfc_kwargs, self.rebuild_existing) for t_id in tracks.index),
            total=len(tracks),
        )
        track_x, track_y = [], []
        for processed_track in processed_tracks:
            if processed_track is None:
                continue
            track_id, track_path = processed_track
            genre_code = genres[track_id]
            track_x.append(track_path)
            track_y.append(genre_code)

        self.track_x = np.array(track_x)
        self.track_y = np.array(track_y)
        assert len(self.track_x) == len(self.track_y)
        return n_genres

    def build_features_for_track_file(self, track_filename):
        """Convenience wrapper around _build_track_features for the features of one mp3 file"""
        return _build_features_for_file(track_filename, self.mfc_kwargs)

    def build_features_for_track_id(self, track_id):
        """Convenience wrapper around _build_track_features for the features of one track"""
        track_path = utils.get_audio_path(AUDIO_DIR, track_id)
        return self.build_features_for_track_file(track_path)

    def setup(self, stage=None):
        # do splits, transforms, parameter-dependent stuff,
        # set relevant variables
        n_tracks = len(self.track_x)
        train_frac, val_frac, test_frac = 0.8, 0.1, 0.1

        X_train, X_test, y_train, y_test = train_test_split(
            self.track_x, self.track_y, test_size=test_frac, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_frac / train_frac, random_state=42
        )

        if stage == "fit" or stage is None:
            self.train = FMASplit(X_train, y_train)
            self.val = FMASplit(X_val, y_val)
        if stage == "test" or stage is None:
            self.test = FMASplit(X_test, y_test)

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

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        # get one item from data source by index
        # probably call ToTensor on the data
        X_path = self.X[index]
        try:
            loaded_X = np.load(X_path)["X"]
            y = self.y[index]
            if loaded_X.shape[-1] != 1280:
                print(X_path, loaded_X.shape)
            return torch.FloatTensor(loaded_X), np.int64(y)
        except Exception as e:
            print(f"Broke on {X_path}")
            raise e

    def __len__(self):
        return len(self.X)

