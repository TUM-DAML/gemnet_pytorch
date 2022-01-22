import functools
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import (
    BatchSampler,
    SubsetRandomSampler,
    SequentialSampler,
)

def collate(batch, target_keys):
    """
    custom batching function because batches have variable shape
    """
    batch = batch[0]  # already batched: Batching happens in DataContainer
    inputs = {}
    targets = {}
    for key in batch:
        if key in target_keys:
            targets[key] = batch[key]
        else:
            inputs[key] = batch[key]
    return inputs, targets

class DataProvider:
    """
    Parameters
    ----------
        data_container: DataContainer
            Contains the dataset.
        ntrain: int
            Number of samples in the training set.
        nval: int
            Number of samples in the validation set.
        batch_size: int
            Number of samples to process at once.
        seed: int
            Seed for drawing samples into train and val set (and shuffle).
        random_split: bool
            If True put the samples randomly into the subsets else in order.
        shuffle: bool
            If True shuffle the samples after each epoch.
        sample_with_replacement: bool
            Sample data from the dataset with replacement.
        split: str/dict
            Overwrites settings of 'ntrain', 'nval', 'random_split' and 'sample_with_replacement'.
            If of type dict the dictionary is assumed to contain the index split of the subsets. 
            If split is of type str then load the index split from the .npz-file.
            Dict and split file are assumed to have keys 'train', 'val', 'test'. 
    """

    def __init__(
        self,
        data_container,
        ntrain: int,
        nval: int,
        batch_size: int = 1,
        seed: int = None,
        random_split: bool = False,
        shuffle: bool = True,
        sample_with_replacement: bool = False,
        split = None,
        **kwargs
    ):
        self.kwargs = kwargs
        self.data_container = data_container
        self._ndata = len(data_container)
        self.batch_size = batch_size
        self.seed = seed
        self.random_split = random_split
        self.shuffle = shuffle
        self.sample_with_replacement = sample_with_replacement

        # Random state parameter, such that random operations are reproducible if wanted
        self._random_state = np.random.RandomState(seed=seed)

        if split is None:
            self.nsamples, self.idx = self._random_split_data(ntrain, nval)
        else:
            self.nsamples, self.idx = self._manual_split_data(split)

    def _manual_split_data(self, split):

        if isinstance(split, (dict,str)):
            if isinstance(split, str):
                # split is the path to the file containing the indices
                assert split.endswith(".npz") , "'split' has to be a .npz file if 'split' is of type str"
                split = np.load(split)

            keys = ["train", "val", "test"]
            for key in keys:
                assert key in split.keys(), f"{key} is not in {[k for k in split.keys()]}"

            idx = {key: np.array(split[key]) for key in keys}
            nsamples = {key: len(idx[key]) for key in keys}

            return nsamples, idx

        else:
            raise TypeError("'split' has to be either of type str or dict if not None.")

    def _random_split_data(self, ntrain, nval):

        nsamples = {
            "train": ntrain,
            "val": nval,
            "test": self._ndata - ntrain - nval,
        }

        all_idx = np.arange(self._ndata)
        if self.random_split:
            # Shuffle indices
            all_idx = self._random_state.permutation(all_idx)

        if self.sample_with_replacement:
            # Sample with replacement so as to train an ensemble of Dimenets
            all_idx = self._random_state.choice(all_idx, self._ndata, replace=True)

        # Store indices of training, validation and test data
        idx = {
            "train": all_idx[0:ntrain],
            "val": all_idx[ntrain : ntrain + nval],
            "test": all_idx[ntrain + nval :],
        }

        return nsamples, idx

    def save_split(self, path):
        """
        Save the split of the samples to path.
        Data has keys 'train', 'val', 'test'.
        """
        assert isinstance(path, str)
        assert path.endswith(".npz"), "'path' has to end with .npz"
        np.savez(path, **self.idx)

    def get_dataset(self, split, batch_size=None):
        assert split in self.idx
        if batch_size is None:
            batch_size = self.batch_size
        shuffle = self.shuffle if split == "train" else False

        indices = self.idx[split]
        if shuffle:
            torch_generator = torch.Generator()
            if self.seed is not None:
                torch_generator.manual_seed(self.seed)
            idx_sampler = SubsetRandomSampler(indices, torch_generator)
            dataset = self.data_container 
        else:
            subset = Subset(self.data_container, indices)
            idx_sampler = SequentialSampler(subset)
            dataset = subset 

        batch_sampler = BatchSampler(
            idx_sampler, batch_size=batch_size, drop_last=False
        )

        dataloader = DataLoader(
            dataset,
            sampler=batch_sampler,
            collate_fn=functools.partial(collate, target_keys=self.data_container.targets),
            pin_memory=True,  # load on CPU push to GPU
            **self.kwargs
        )

        # loop infinitely
        # we use the generator as the rest of the code is based on steps and not epochs
        def generator():
            while True:
                for inputs, targets in dataloader:
                    yield inputs, targets

        return generator()
