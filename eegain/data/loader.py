import logging
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import EEGDataset, EEGDatasetBase

logger = logging.getLogger("Dataloader")
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)


class EEGDataloader:
    def __init__(self, dataset: EEGDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def loto(self, subject_id, session_ids, n_fold):
        fold_size = len(session_ids) // n_fold

        folds = [session_ids[i:i + fold_size] for i in range(0, len(session_ids), fold_size)]
        for test_sessions in folds:
            logger.debug(f"subject_id is: {subject_id}, test sessions are: {test_sessions}")
            train_sessions = session_ids.copy()
            train_sessions = [item for item in session_ids if item not in test_sessions]
            logger.debug(f"subject_id is: {subject_id}, train sessions are: {train_sessions}")
            test_data = self.dataset.__get_trials__(test_sessions, subject_id)
            train_data = self.dataset.__get_trials__(train_sessions, subject_id)
            train_data, train_label, _ = EEGDataloader._concat_data(train_data, loader_type="LOTO")
            test_data, test_label, _ = EEGDataloader._concat_data(test_data, loader_type="LOTO")

            if len(train_data.shape) != 4:  # DREAMER has already shape that is needed and it doesn't need normalization
                train_data, test_data = EEGDataloader.normalize(train_data, test_data)

            train_dataloader = self._get_dataloader(train_data, train_label)
            test_dataloader = self._get_dataloader(test_data, test_label)
            yield {
                "train": train_dataloader,
                "test": test_dataloader,
                "test_session_indexes": test_sessions,
                "train_session_indexes": train_sessions,
                "subject_id": subject_id,
            }

    @staticmethod
    def _concat_data(
        data: List[Tuple[Dict[int, np.ndarray], Dict[int, int]]]
    , loader_type="LOSO") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        concatenate data from several subjects and make one tensor

        Args:
            data: list of subjects data and label. each value is a dictionary with session and data pairs
            we just need values for training and not session ids

        Returns:
            x,y (torch.Tensor):
        """
        
        if loader_type == "LOTO":
            x = np.concatenate([v for v in data[0].values()], axis=0)
        else:
            x = np.concatenate([v for d in data for v in d[0].values()], axis=0)
        x = torch.from_numpy(x).float()

        # TODO: refactor
        y = []
        video_ids = []
        if loader_type == "LOTO":
            for i in data[1].items():
                k = i[0]
                v = i[1]
                y.extend(list(np.repeat(v, data[0][k].shape[0])))
            y = torch.tensor(y) 
        else:
            for i in data:
                #i is each subject
                for k, v in i[1].items():
                    #k - clip index
                    y.extend(list(np.repeat(v, i[0][k].shape[0])))
                    video_ids.extend(list(np.repeat(k, i[0][k].shape[0])))
            y = torch.tensor(y)


        return x, y,video_ids

    def _get_dataloader(
        self, data: torch.Tensor, label: torch.Tensor, shuffle: bool = True #changed to False
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
        if len(train_data.shape) != 4:
          train_data = train_data.unsqueeze(1)
          test_data = test_data.unsqueeze(1)
        for channel in range(train_data.shape[2]):
            #this is for important for amigos and maybe dreame
            #TODO: why do they have nan values???
            nan_mask = torch.isnan(train_data[:, :, channel, :])  
            train_data[:, :, channel, :][nan_mask] = 0

            nan_mask = torch.isnan(test_data[:, :, channel, :])  
            test_data[:, :, channel, :][nan_mask] = 0
            
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
            #print(train_data)
            train_data, train_label, train_videos = EEGDataloader._concat_data(train_data)
            logger.debug(f"train data shape {train_data.shape}")

            logger.debug(f"Preparing: test subjects: {test_subject_ids}")
            test_data = [self.dataset.__get_subject__(i) for i in test_subject_ids]
            test_data, test_label, test_videos = EEGDataloader._concat_data(test_data)
            logger.debug(f"test data shape {test_data.shape}")

            train_data, test_data = EEGDataloader.normalize(train_data, test_data)
            print(f"Number of train samples: {len(train_label)}, Number of test samples: {len(test_label)}")

            train_dataloader = self._get_dataloader(train_data, train_label) #changed to False
            test_dataloader = self._get_dataloader(test_data, test_label) #changed to False
            yield {
                "train": train_dataloader,
                "test": test_dataloader,
                "test_subject_indexes": test_subject_ids,
                "train_subject_indexes": train_subject_ids,
                "train_videos": train_videos,
                "test_videos":test_videos,
            }

    def loso_fixed(self, train_subject_ids, test_subject_ids) -> Dict[str, Any]:
        logger.info(f"Splitting type: leave-n-subject-out-fixed")

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
        return {
            "train": train_dataloader,
            "test": test_dataloader,
            "test_subject_indexes": test_subject_ids,
            "train_subject_indexes": train_subject_ids,
        }
