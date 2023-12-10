import logging
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import EEGDataset, EEGDatasetBase

logger = logging.getLogger("Dataloader")


class EEGDataloader:
    def __init__(self, dataset: EEGDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def loto(self):
        pass

    @staticmethod
    def _concat_data(
        data: List[Tuple[Dict[int, np.ndarray], Dict[int, int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        concatenate data from several subjects and make one tensor

        Args:
            data: list of subjects data and label. each value is a dictionary with session and data pairs
            we just need values for training and not session ids

        Returns:
            x,y (torch.Tensor):
        """
        x = np.concatenate([v for d in data for v in d[0].values()], axis=0)
        x = torch.from_numpy(x).float()

        # TODO: refactor
        y = []
        for i in data:
            for k, v in i[1].items():
                y.extend(list(np.repeat(v, i[0][k].shape[0])))
        y = torch.tensor(y)

        return x, y

    def _get_dataloader(
        self, data: torch.Tensor, label: torch.Tensor, shuffle: bool = True
    ) -> DataLoader:
        """
        Returns a DataLoader object that can be used for iterating through batches of data and labels.

        Args:
            data (List[torch.Tensor]): List of input data tensors.
            label (List[torch.Tensor]): List of label tensors.
            shuffle (bool): Whether to shuffle the data before batching. Defaults to True.

        Returns:
            loader: A DataLoader object that can be used to iterate through batches of data and labels.
        """
        dataset = EEGDatasetBase(data, label)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )
        return loader

    @staticmethod
    def normalize(
        train_data: torch.Tensor, test_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        this function do standard normalization for EEG channel by channel
        Args:
            train_data (torch.Tensor): training data
            test_data (torch.Tensor): testing data

        Returns:
            normalized training and testing data
        """
        # data: sample x 1 x channel x data
        for channel in range(train_data.shape[2]):
            std, mean = torch.std_mean(train_data[:, :, channel, :])
            train_data[:, :, channel, :] = (train_data[:, :, channel, :] - mean) / std
            test_data[:, :, channel, :] = (test_data[:, :, channel, :] - mean) / std
        return train_data, test_data
    
    def get_subject(self, idx, num_test_rec):
        subject_data = self.dataset.__get_subject__(idx)

        session_idxs = list(subject_data[1].keys())
        training_idxs = session_idxs[ : -num_test_rec]
        testing_idxs  = session_idxs[- num_test_rec :]

        training_sessions = {}
        training_labels = {}

        testing_sessions = {}
        testing_labels = {}

        for idx in training_idxs:
            training_sessions[idx] = subject_data[0][idx]
            training_labels[idx] = subject_data[1][idx]

        for idx in testing_idxs:
            testing_sessions[idx] = subject_data[0][idx]
            testing_labels[idx] = subject_data[1][idx]

        train_data, train_label = EEGDataloader._concat_data([(training_sessions, training_labels)])
        test_data, test_label = EEGDataloader._concat_data([(testing_sessions, testing_labels)])

        train_data, test_data = EEGDataloader.normalize(train_data, test_data)

        train_dataloader = self._get_dataloader(train_data, train_label)
        test_dataloader = self._get_dataloader(test_data, test_label)
        return {
            "train": train_dataloader,
            "test": test_dataloader,
            "test_subject_indexes": [idx],
            "train_subject_indexes": [idx],
        }

    def loso(self, subject_out_num: int = 1) -> Dict[str, Any]:
        logger.info(f"Splitting type: leave-one-subject-out")
        subject_ids = self.dataset.__get_subject_ids__()
        test_ids_combination = list(combinations(subject_ids, subject_out_num))

        for test_subject_ids in test_ids_combination:
            train_subject_ids = set(subject_ids) - set(test_subject_ids)

            logger.debug(f"Preparing: train subjects: {train_subject_ids}")
            train_data = [self.dataset.__get_subject__(i) for i in train_subject_ids]
            train_data, train_label = EEGDataloader._concat_data(train_data)
            logger.debug(f"train data shape {train_data.shape}")

            logger.debug(f"Preparing: test subjects: {test_subject_ids}")
            test_data = [self.dataset.__get_subject__(i) for i in test_subject_ids]
            test_data, test_label = EEGDataloader._concat_data(test_data)
            logger.debug(f"test data shape {test_data.shape}")

            train_data, test_data = EEGDataloader.normalize(train_data, test_data)

            train_dataloader = self._get_dataloader(train_data, train_label)
            test_dataloader = self._get_dataloader(test_data, test_label)
            yield {
                "train": train_dataloader,
                "test": test_dataloader,
                "test_subject_indexes": test_subject_ids,
                "train_subject_indexes": train_subject_ids,
            }
