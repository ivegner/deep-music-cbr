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
USE_FMA_SMALL = os.environ.get("USE_SMALL")
if USE_FMA_SMALL:
    AUDIO_DIR = os.environ.get("AUDIO_DIR_SMALL")
else:
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
    if USE_FMA_SMALL: # FMA small has a different number of classes, so the y will be different.
        param_string = "small_" + param_string
    path = os.path.join(PREPPED_DATA_DIR, tid_str[:3], tid_str)
    return os.path.join(path, param_string + ".npz")


def _build_track_features(k):
    """
    Build numpy array of numerical features for a given track
    """
    # https://medium.com/@tanveer9812/mfccs-made-easy-7ef383006040
    # https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
    track_id, track_info, mfc_kwargs, rebuild_existing = k
    try:
        track_data_path = _get_track_data_path(track_id, mfc_kwargs)
        if os.path.exists(track_data_path):
            if rebuild_existing:
                # delete in case the data was bad, will be rebuilt anyway if good
                os.remove(track_data_path)
            else:
                # append and skip
                return track_data_path
        # comment claimed that this function doesn't work correctly
        track_filename = utils.get_audio_path(AUDIO_DIR, track_id)
        # print(f"Processing {i}/{len(tracks)}, {track_filename=}", end="\r")
        with warnings.catch_warnings():
            # raises "UserWarning: PySoundFile failed. Trying audioread instead."
            # see https://github.com/librosa/librosa/issues/1015
            warnings.simplefilter("ignore")
            # load song audio and sample rate.
            # duration is fixed because some clips are just a little shorter
            DURATION_CLIP = 29.5
            audio_data, sample_rate = librosa.load(
                track_filename, sr=mfc_kwargs["resample_rate"], mono=True, duration=DURATION_CLIP,
            )
            if audio_data.shape[0] != DURATION_CLIP * mfc_kwargs["resample_rate"]:
                print(
                    f"Track {track_id} has duration {audio_data.shape[0]/mfc_kwargs['resample_rate']}, not {DURATION_CLIP}. Rejecting.",
                    flush=True,
                )
                return
        track_x = _build_mfc(audio_data, sample_rate, mfc_kwargs)
        track_x = np.array(track_x, dtype=float)
        # save data
        _save_track_data(track_data_path, track_x, track_info)
        # print(track_id, track_x.shape, track_info.shape, track_filename, sample_rate, flush=True)
        return track_data_path
    except Exception as e:
        print(f"Track {track_id} broke with error {e}", flush=True)
        traceback.print_exc()
        return


def _save_track_data(data_path, track_features, track_y):
    """
    Save track features to PREPPED_DATA_DIR, to be loaded later
    """
    data_dir = os.path.dirname(data_path)
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(data_path, X=track_features, y=track_y)


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
        self,
        autoencode=False,
        use_echonest=False,
        batch_size=64,
        rebuild_existing=False,
        mfc_kwargs=None,
        num_workers=1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.autoencode = autoencode
        self.use_echonest = use_echonest
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

        if USE_FMA_SMALL:
            tracks = tracks[tracks['set', 'subset'] <= 'small']
            # remove genres with no tracks (should be 8 genres resulting)
            tracks[("track", "genre_top")] = tracks[("track", "genre_top")].cat.remove_unused_categories()

        n_genres = tracks[("track", "genre_top")].nunique()

        ###### Find Echonest Tracks ######
        interesting_base_cols = ["genre_top"]  # , "genres"]
        # isolate tracks with echonest data
        # prune columns we don't want to train on (from the non-echonest data)

        # If we are using the full echonest data
        if self.use_echonest:
            echonest = utils.load("data/fma_metadata/echonest.csv")
            if n_subset is not None:
                echonest = echonest.head(n_subset)

            tracks = tracks.loc[
                echonest.index, tracks.columns.get_level_values(1).isin(interesting_base_cols)
            ]
            # join with echonest information. Can also grab song hotttness if interested.
            tracks = tracks["track"].join(echonest[("echonest", "audio_features")], how="inner")

            # scale tempo
            self.tempo_scaler = StandardScaler()
            tracks["tempo"] = self.tempo_scaler.fit_transform(tracks[["tempo"]])
        else:
            tracks = tracks[[("track", "genre_top")]]

        ###### Clean and Prep Labels ######
        # clean songs with NA values (only genre can be missing).
        print("NA values per feature:", tracks.isna().sum(), sep="\n")
        tracks = tracks.dropna()
        print(f"Total clean tracks: {len(tracks)}")
        print("Genre counts:", tracks.groupby([("track", "genre_top")]).size(), sep="\n")
        # One-hot encode genres
        tracks = pd.get_dummies(tracks, columns=[("track", "genre_top")], prefix=["genre_is"])
        n_features = len(tracks.columns)
        if not self.use_echonest: assert n_genres == n_features
        print(f"Total {n_features} features ({n_genres} of them genres)")

        # # incorporate track_id into columns and reset index to numeric
        # tracks = tracks.reset_index(drop=False)

        ###### Build Features ######

        # hack workaround to avoid rebuilding features 
        # to un-hack it, keep just the first kwargs=... line and 
        # delete the prepped data directory
        if not self.use_echonest:
            kwargs = dict(**self.mfc_kwargs, echonest=self.use_echonest)
        else:
            kwargs = self.mfc_kwargs
        self.track_paths = [
            t
            for t in process_map(
                _build_track_features,
                (
                    (t_id, t_row, kwargs, self.rebuild_existing)
                    for t_id, t_row in tracks.iterrows()
                ),
                total=len(tracks),
            )
            if t is not None
        ]
        return n_features, n_genres

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
        if stage == "fit" or stage is None:
            self.train = FMASplit(train_paths)
            self.val = FMASplit(val_paths)
        if stage == "test" or stage is None:
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
        try:
            loaded = np.load(self.data_paths[index])
            X, y = loaded["X"], loaded["y"]
            if self.autoencode:
                y = X
            if X.shape[-1] != 1271:
                print(self.data_paths[index], X.shape)
            return torch.tensor(X).float(), torch.tensor(y).float()
        except Exception as e:
            print(f"Broke on {self.data_paths[index]}")
            raise e

    def __len__(self):
        return self.count

