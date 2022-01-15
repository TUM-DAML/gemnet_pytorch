import functools
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def collate(batch, target_keys):
    """
    custom batching function because batches have variable shape
    """
    batch = batch[0]  # already batched
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
        transforms: list
            List of transformations applied to the dataset.
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
        **kwargs
    ):
        self.data_container = data_container
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.kwargs = kwargs

        # Random state parameter, such that random operations are reproducible if wanted
        _random_state = np.random.RandomState(seed=seed)

        all_idx = np.arange(len(data_container))
        if random_split:
            # Shuffle indices
            all_idx = _random_state.permutation(all_idx)

        if sample_with_replacement:
            # Sample with replacement so as to train an ensemble of Dimenets
            all_idx = _random_state.choice(all_idx, len(all_idx), replace=True)

        # Store indices of training, validation and test data
        self.idx = {
            "train": all_idx[0:ntrain],
            "val": all_idx[ntrain : ntrain + nval],
            "test": all_idx[ntrain + nval :],
        }

    def get_dataset(self, split, batch_size=None):
        assert split in self.idx
        if batch_size is None:
            batch_size = self.batch_size
        shuffle = self.shuffle if split == "train" else False
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        else:
            generator = None

        dataloader = DataLoader(
            Subset(self.data_container, self.idx[split]),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=functools.partial(collate, target_keys=data_container.targets),
            pin_memory=True,  # load on CPU, push to GPU
            **self.kwargs
        )

        # loop infinitely
        # we use the generator as the rest of the code is based on steps and not epochs
        def generator():
            while True:
                for inputs, targets in dataloader:
                    yield inputs, targets

        return generator()
