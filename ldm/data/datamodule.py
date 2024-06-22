import pytorch_lightning as pl
import numpy as np
import torch

from torch.utils.data import IterableDataset, random_split, DataLoader, Dataset, Subset
from functools import partial
from typing import List
from omegaconf import ListConfig, DictConfig

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.misc.util import instantiate_from_config, get_obj_from_str

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


def collate_fn_diffmap(batch: List) -> dict:
    # Keys to handle manually.
    ignored_keys = ['camera_ctxt', 'camera_trgt']

    # Default batch preprocessing
    collate_batch = [{key: value for key, value in sample.items() if key not in ignored_keys} for sample in batch]
    collate_batch = torch.utils.data.dataloader.default_collate(collate_batch)

    # Stack other fields in lists.
    for key in ignored_keys:
        if key in batch[0]:
            collate_batch[key] = [sample[key] for sample in batch]

    return collate_batch


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train: DictConfig | None = None,
        validation: DictConfig | None = None,
        test: DictConfig | None = None,
        predict: DictConfig | None = None,
        num_workers: int | None = None,
        num_val_workers: int | None = None,
        use_worker_init_fn: bool = False,
        collate_fn: str | None = None,
        wrap: bool = False,
        shuffle_val_dataloader: bool = False,
        shuffle_test_loader: bool = False,
    ) -> None:
        
        super().__init__()

        self.batch_size = batch_size

        # Set workers.
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if num_val_workers is None:
            self.num_val_workers = self.num_workers
        else:
            self.num_val_workers = num_val_workers
        self.use_worker_init_fn = use_worker_init_fn

        # Set datasets configs and set dataloaders instatiators.
        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader

        if validation is not None:
            self.dataset_configs["validation"] = validation
            if self.dataset_configs["validation"].get('params', {}).get('val_scenes', []) is not None:
                self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        
        # Use custom collate function.
        if collate_fn is not None:
            self.collate_fn = get_obj_from_str(collate_fn)
        
        self.wrap = wrap

    def prepare_data(self) -> None:
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None) -> None:
        # Instantiate datasets.
        self.datasets = {
            k: instantiate_from_config(self.dataset_configs[k])
            for k in self.dataset_configs
        }

        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self) -> DataLoader:
        # Set workers initialization.
        is_iterable_dataset = isinstance(self.datasets['train'], IterableDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # Create train loader.
        shuffle = False if is_iterable_dataset else True # disabled for iterable dataset
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn
        )

    def _val_dataloader(self, shuffle=False) -> DataLoader:
        if isinstance(self.datasets['validation'], IterableDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        
        # Create val loader.
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_val_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle, collate_fn=self.collate_fn
        )

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], IterableDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # Create test loader.
        shuffle = shuffle and (not is_iterable_dataset) # disabled for iterable dataset
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], IterableDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            collate_fn=self.collate_fn
        )