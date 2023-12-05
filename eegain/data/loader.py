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

    def loto(self, video_out_num=1):
        logger.info(f"Splitting type: leave-one-trial-out")
        subject_ids = self.dataset.__get_subject_ids__()
        num_trials = 20  # TODO this is correct only for mahnob
        video_idxs = set(i for i in range(1, num_trials + 1))
        test_video_trials_combination = list(combinations(video_idxs, video_out_num))

        for subject_id in subject_ids:
            for test_video_trial in test_video_trials_combination:
                # TODO get_subjects and get_subject_videos are similar to each other
                data = self.dataset.__get_subject_videos__(subject_id, video_out_idx=test_video_trial)
                train_data = [data[0]]
                test_data = [data[1]]
                train_data, train_label = EEGDataloader._concat_data(train_data)
                test_data, test_label = EEGDataloader._concat_data(test_data)

                train_data, test_data = EEGDataloader.normalize(train_data, test_data)

                train_dataloader = self._get_dataloader(train_data, train_label)
                test_dataloader = self._get_dataloader(test_data, test_label)
                yield {
                    "train": train_dataloader,
                    "test": test_dataloader,
                    "test_video_indexes": test_video_trial,
                    "train_video_indexes": video_idxs - set(test_video_trial),
                }

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
