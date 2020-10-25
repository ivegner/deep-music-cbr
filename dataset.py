from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset
import pandas as pd
import pytorch_lightning as pl


class MusicDataModule(pl.LightningDataModule):
    def __init__(self, spectrogram_dims=128, batch_size=64):
        super().__init__()
        self.spectrogram_dims = spectrogram_dims #example
        self.batch_size = batch_size

    def prepare_data(self):
        # called only on 1 GPU
        # do things that are only done once here
        # if data isn't downloaded, download data
        # save data on disk
        # load data into some big Pandas object
        pass


    def setup(self):
        # do splits, transforms, parameter-dependent stuff,
        # set relevant variables
        # e.g.
        # build_spectrograms(data, self.spectrogram_dims)
        # self.train, self.val, self.test = split_data()
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


"""
# guide on building a dataset: https://github.com/utkuozbulak/pytorch-custom-dataset-examples
class MusicDataset(Dataset):
    '''
    Music dataset sourced from FMA
    '''

    def __init__(self):
        # if data isn't downloaded, download data
        # save data
        # load data into Pandas
        # prep data
        # set data source object, make sure __len__ can get its length
        pass

    def __getitem__(self, index):
        # get one item from data source by index
        # probably call ToTensor on the data
        return (img, label)

    def __len__(self):
        return count  # of how many examples(images?) you have
 """