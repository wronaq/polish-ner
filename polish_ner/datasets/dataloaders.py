import torch
from torch.utils.data import DataLoader, Subset


class DataLoaders:
    def __init__(self, train_dataset, valid_dataset, test_dataset):

        self._datasets = {
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset,
        }
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self._testing_batch = None

        self._make_dataloaders()
        self._prepare_testing_batch()

    def _make_dataloaders(self):
        self.train_loader = DataLoader(
            dataset=self._datasets["train"],
            batch_size=self._datasets["train"].batch_size,
            shuffle=True,
            num_workers=self._datasets["train"].num_workers,
        )
        self.valid_loader = DataLoader(
            dataset=self._datasets["valid"],
            batch_size=self._datasets["valid"].batch_size,
            shuffle=True,
            num_workers=self._datasets["valid"].num_workers,
        )
        self.test_loader = DataLoader(
            dataset=self._datasets["test"],
            batch_size=self._datasets["test"].batch_size,
            shuffle=False,
            num_workers=self._datasets["test"].num_workers,
        )

        return self.train_loader, self.valid_loader, self.test_loader

    def _prepare_testing_batch(self):
        # preparing small deterministic batch for model testing
        self._testing_batch = DataLoader(
            dataset=Subset(self._datasets["train"], torch.arange(0, 8)),
            batch_size=8,
            shuffle=False,
            num_workers=self._datasets["train"].num_workers,
        )
