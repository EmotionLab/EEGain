import logging
import os
import re
import pickle
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.utils.data import Dataset

from ..transforms import Construct

logger = logging.getLogger("Dataset")


class EEGDatasetBase(Dataset):
    """
    Python dataset wrapper that takes tensors and implements dataset structure
    """

    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor):
        self.x = x_tensor
        self.y = y_tensor

        assert len(self.x) == len(self.y)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.y)


class EEGDataset(ABC):
    @abstractmethod
    def __get_subject_ids__(self) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def __get_subject__(
        self, subject_index: int
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        raise NotImplementedError


class SeedIV(EEGDataset):
    @staticmethod
    def _create_user_mat_mapping(data_path: Path) -> Dict[int, List[str]]:
        """This method creates mapping between users and user_recordings
        Args:
            data_path(Path): path to SEED IV dataset

        Returns:
            user_session_info(Dict[int, List[str]]): This is the dictionary where key is user_id and value is list
                                                     of file_names that's associated to this particular user
        """

        num_sessions = 3  # There are three sessions in SEED IV dataset
        user_session_info: Dict[int, List[str]] = defaultdict(list)

        for session in range(1, num_sessions + 1):
            path = (
                data_path / Path("eeg_raw_data") / Path(str(session)) # eeg_raw_data
            )  # Path to particular sessions mat files
            for mat_file_path in path.glob("*.mat"):
                subject_file_name = mat_file_path.name
                subject_id = int(
                    subject_file_name[: subject_file_name.index("_")]
                )  # file name starts with user_id
                user_session_info[subject_id].append(
                    str(session) + "/" + subject_file_name
                )

        return user_session_info

    def __init__(self, root: str, label_type: str, transform: Construct = None):
        """This is just constructor for SEED IV class
        Args:
            root(str): Path to SEED IV dataset folder
            label_type(str): 'V' for Valence and 'A' for Arousal
            transform(Construct): transformations to apply on dataset

        Return:
        """

        self.root = Path(root)
        self.transform = transform
        self.mapping_list = SeedIV._create_user_mat_mapping(self.root)
        self.label_type = label_type
        self._num_trials = 24
        self._sampling_rate = 1000

    def __get_subject_ids__(self) -> List[int]:
        """Method returns list of subject ids"""

        return list(self.mapping_list.keys())

    def __get_subject__(
        self, subject_index: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """This method returns data and associated labels for specific subject

        Args:
            subject_index(int): user id

        Returns:
            data_array(Dict[str, np.ndarray]): Dictionary of files and data associated to specific user
            label_array(Dict[str, int]): labels for each recording
        """

        labels_file_path = self.root / Path("ReadMe.txt")
        path_to_channels_excel = self.root / Path("Channel Order.xlsx")
        channels_file = pd.read_excel(path_to_channels_excel, header=None)
        channels = list(channels_file.iloc[:, 0])

        num_trials = self._num_trials
        sampling_rate = self._sampling_rate
        sessions = self.mapping_list[subject_index]
        data_array, label_array = {}, {}

        for session in sessions:
            path_to_mat = self.root / Path("eeg_raw_data") / Path(str(session)) # eeg_raw_data
            mat_data = scipy.io.loadmat(path_to_mat)  # Get Matlab File
            mat_data_values = list(mat_data.values())[3:]  # Matlab file contains some not necessary info so let's remove it
            for trial in range(1, num_trials + 1):
                eeg_data = mat_data_values[
                    trial - 1
                ]  # each matlab file contains 24 trials. Here we take one trial
                info = mne.create_info(
                    ch_names=channels, sfreq=sampling_rate, ch_types="eeg"
                )
                raw_data = mne.io.RawArray(
                    eeg_data, info, verbose=False
                )  # convert numpy ndarray to mne object

                if self.transform:  # apply transformations
                    raw_data = self.transform(raw_data)

                session_trial = session + "/" + str(trial)
                data_array[session_trial] = raw_data.get_data()

                with open(labels_file_path, "r") as file:
                    labels_file_content = file.read()

                # Extract label from file and add to label_array
                session_id = session[: session.index("/")]
                session_start_idx = labels_file_content.index(
                    f"session{session_id}_label"
                )
                session_end_idx = labels_file_content.index(";", session_start_idx)
                session_labels = labels_file_content[session_start_idx:session_end_idx]
                session_labels = eval(session_labels[session_labels.index("[") :])

                emotional_label = session_labels[trial - 1]
                label_array[session_trial] = emotional_label

        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array


class MAHNOB(EEGDataset):
    @staticmethod
    def _create_user_csv_mapping(data_path: Path) -> Dict[int, List[str]]:
        """
        this functions creates mapping dictionary {user_id: [all csv files names for this user_id]}
        """
        user_session_info: Dict[int, List[str]] = defaultdict(list)

        for xml_file_path in data_path.glob("**/*.xml"):
            with open(xml_file_path, "r") as f:
                data = f.read()
                root = ET.fromstring(data)
            subj_id = int(root.find("subject").attrib["id"])
            session_id = root.attrib["sessionId"]
            user_session_info[subj_id].append(session_id)

        logger.debug(f"Subject id -> sessions: {user_session_info}")
        return user_session_info

    def __init__(self, root: str, label_type: str, transform: Construct = None):
        self.root = Path(root)
        self.transform = transform
        self.mapping_list = MAHNOB._create_user_csv_mapping(self.root)
        self.label_type = label_type

        logger.info(f"Using Dataset: {self.__class__.__name__}")
        logger.info(f"Using label: {self.label_type}")

    def __get_subject_ids__(self) -> List[int]:
        return list(self.mapping_list.keys())

    def __get_subject__(
        self, subject_index: int
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        sessions = self.mapping_list[subject_index]

        data_array, label_array = {}, {}

        for session in sessions:
            xml_path = next((Path(self.root) / session).glob("*.xml"), None)
            bdf_path = next((Path(self.root) / session).glob("*.bdf"), None)
            if (xml_path is None) or (bdf_path is None):
                continue

            data = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)

            if self.transform:
                data = self.transform(data)

            data_array[session] = data.get_data()

            with open(xml_path, "r") as f:
                data = f.read()
                root = ET.fromstring(data)
            felt_arousal = int(root.attrib["feltArsl"])
            felt_valence = int(root.attrib["feltVlnc"])
            label_array[session] = np.array([felt_arousal, felt_valence])

        # choose label and split into binary
        idx = 0 if self.label_type == "A" else 1
        for session_id, data in label_array.items():
            target_label = data[idx]
            target_label = 0 if target_label <= 4.5 else 1
            label_array[session_id] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        logger.debug(
            f"subj index: {subject_index} data {data_array.keys()}, label {label_array.keys()}"
        )
        return data_array, label_array

    def __get_videos__(self, session_ids: List, subject_video_mapping = None):
        data_array, label_array = {}, {}
        for session_id in session_ids:

            xml_path = next((Path(self.root) / session_id).glob("*.xml"), None)
            bdf_path = next((Path(self.root) / session_id).glob("*.bdf"), None)

            data = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)

            if self.transform:
                data_bdf = self.transform(data)

            with open(xml_path, "r") as f:
                data = f.read()
                root = ET.fromstring(data)

            felt_arousal = int(root.attrib["feltArsl"])
            felt_valence = int(root.attrib["feltVlnc"])

            data_array[session_id] = data_bdf.get_data()
            label_array[session_id] = np.array([felt_arousal, felt_valence])

        # choose label and split into binary
        idx = 0 if self.label_type == "A" else 1
        for session_id, data in label_array.items():
            target_label = data[idx]
            target_label = 0 if target_label <= 5 else 1
            label_array[session_id] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array


class DEAP(EEGDataset):
    @staticmethod
    def _create_user_csv_mapping(
        file_paths: List[str], data_path: Path, transform, preprocessed: bool = True
    ) -> Dict[int, List[str]]:
        """
        this functions creates mapping dictionary {user_id: [read file for each subject which contains all sessions]}
        """
        user_session_info = {}
        if preprocessed:
            for subj in file_paths:
                if ".dat" in subj:
                    id = subj[1:-4]
                    with open(os.path.join(data_path, subj), "rb") as file:
                        data = pickle.load(file, encoding="latin-1")
                        if transform:
                            # TODO - this should be done with transform itself but transform doesn't work with
                            # this data because it's read by pickle and type of data['data'] is numpy ndarray
                            data['data'] = data['data'][:, :32, :]
                        user_session_info[id] = data

        return user_session_info

    def __init__(
        self,
        root: str,
        label_type: str,
        preprocessed: bool = True,
        transform: Construct = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.file_paths = os.listdir(self.root)
        self.label_type = label_type
        self.user_mapping = self._create_user_csv_mapping(
            self.file_paths, self.root, self.transform, preprocessed
        )
        self._ch_names = [
            "Fp1",
            "AF3",
            "F3",
            "F7",
            "FC5",
            "FC1",
            "C3",
            "T7",
            "CP5",
            "CP1",
            "P3",
            "P7",
            "PO3",
            "O1",
            "Oz",
            "Pz",
            "Fp2",
            "AF4",
            "Fz",
            "F4",
            "F8",
            "FC6",
            "FC2",
            "Cz",
            "C4",
            "T8",
            "CP6",
            "CP2",
            "P4",
            "P8",
            "PO4",
            "O2",
            "EXG1",
            "EXG2",
            "EXG3",
            "EXG4",
            "GSR1",
            "Resp",
            "Plet",
            "Temp",
        ]

        self._sfreq = 128
        self.info = mne.create_info(
            self._ch_names, self._sfreq, ch_types="misc", verbose=None
        )

    def __get_subject_ids__(self) -> List[int]:
        return self.user_mapping.keys()

    def __get_subject__(
        self, subject_index: int
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        subject_data = self.user_mapping[subject_index]
        datas = subject_data["data"]
        labels = subject_data["labels"]
        data_array, label_array = {}, {}

        for i in range(0, datas.shape[0]):
            data = datas[i]
            data = mne.io.RawArray(data, self.info, verbose=False)
            if self.transform:
                data = self.transform(data)
            data_array[i] = data.get_data()

            felt_arousal = int(labels[i][1])
            felt_valence = int(labels[i][0])
            label_array[i] = np.array([felt_arousal, felt_valence])

        # choose label and split into binary
        idx = 0 if self.label_type == "A" else 1
        for session_id, data in label_array.items():
            target_label = data[idx]
            target_label = 0 if target_label <= 4.5 else 1
            label_array[session_id] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array

    def __get_videos__(self, session_ids: List, subject_video_mapping: Dict):
        data_array, label_array = {}, {}
        for session_id in session_ids:
            curr_labels = subject_video_mapping['labels'][session_id]
            curr_data = subject_video_mapping['data'][session_id]

            felt_valence = curr_labels[0]
            felt_arousal = curr_labels[1]

            data_array[session_id] = curr_data
            label_array[session_id] = np.array([felt_arousal, felt_valence])

        # choose label and split into binary
        idx = 0 if self.label_type == "A" else 1
        for session_id, data in label_array.items():
            target_label = data[idx]
            target_label = 0 if target_label <= 5 else 1
            label_array[session_id] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array


class DREAMER(EEGDataset):
    def __init__(self, root: str, label_type: str, transform: Construct = None):
        """This is just constructor for DREAMER class
        Args:
            root(str): Path to DREAMER.mat file
            label_type(str): 'V' for Valence and 'A' for Arousal
            transform(Construct): transformations to apply on dataset

        Return:
        """
        self.root = Path(root)
        self.transform = transform
        self.label_type = label_type
        self.num_videos = 18
        self.threshold = 3
        self.user_mapping = self._create_user_csv_mapping(
            self.root, self.transform, self.label_type
        )

    def __get_subject_ids__(self) -> List[int]:
        """Method returns list of subject ids"""
        subject_ids = list(range(23))
        return subject_ids

    def __get_subject__(
        self, subject_index: int
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        """This method returns data and associated labels for specific subject

        Args:
            subject_index(int): user id

        Returns:
            data_array(Dict[int, np.ndarray]): Dictionary of files and data associated to specific user
            label_array(Dict[int, int]): labels for each recording
        """

        data_array, label_array = {}, {}

        # choose label
        idx = 4 if self.label_type == "V" else 5

        path_to_dreamer = self.root
        mat_data = scipy.io.loadmat(path_to_dreamer)
        channels = [
            channel_name[0] for channel_name in list(mat_data["DREAMER"][0][0][3])[0]
        ]

        # subject_eeg_full_info contains EEG & ECG recordings of specific person, age, gender and
        # valence/arousal/dominance scores so here is full info about the person
        subject_eeg_full_info = mat_data["DREAMER"][0][0][0][0][subject_index][0][0]

        for video_idx in range(self.num_videos):
            person_eeg_recording = subject_eeg_full_info[2][0][0][1][video_idx][0]
            person_eeg_recording = (
                person_eeg_recording.transpose()
            )  # This is necessary to convert numpy ndarray to mne object
            info = mne.create_info(ch_names=channels, sfreq=128, ch_types="eeg")
            raw_data = mne.io.RawArray(person_eeg_recording, info)

            if self.transform:
                raw_data = self.transform(raw_data)

            data_array[video_idx] = raw_data.get_data()

            target_label = int(subject_eeg_full_info[idx][video_idx][0])
            target_label = 0 if target_label <= self.threshold else 1
            label_array[video_idx] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array

    @staticmethod
    def _create_user_csv_mapping(data_path: Path, transform, label_type) -> Dict[int, List[str]]:
        user_session_info = {}

        mat_data = scipy.io.loadmat(data_path)

        num_subjects = len(list(mat_data["DREAMER"][0][0][0][0]))
        for curr_subject_idx in range(num_subjects):
            num_videos_curr_subject = len(list(mat_data["DREAMER"][0][0][0][0][curr_subject_idx][0][0][2][0][0][1]))
            video_idxs = [i for i in range(num_videos_curr_subject)]
            user_session_info[curr_subject_idx] = video_idxs

        return user_session_info

    def __get_videos__(self, video_ids, subject_id):
        data_array, label_array = {}, {}

        # choose label
        idx = 4 if self.label_type == "V" else 5

        path_to_dreamer = self.root
        mat_data = scipy.io.loadmat(path_to_dreamer)
        channels = [
            channel_name[0] for channel_name in list(mat_data["DREAMER"][0][0][3])[0]
        ]

        # subject_eeg_full_info contains EEG & ECG recordings of specific person, age, gender and
        # valence/arousal/dominance scores so here is full info about the person
        subject_eeg_full_info = mat_data["DREAMER"][0][0][0][0][subject_id][0][0]

        for video_idx in video_ids:
            person_eeg_recording = subject_eeg_full_info[2][0][0][1][video_idx][0]
            person_eeg_recording = (
                person_eeg_recording.transpose()
            )  # This is necessary to convert numpy ndarray to mne object
            info = mne.create_info(ch_names=channels, sfreq=128, ch_types="eeg")
            raw_data = mne.io.RawArray(person_eeg_recording, info)

            if self.transform:
                raw_data = self.transform(raw_data)

            data_array[video_idx] = raw_data.get_data()

            target_label = int(subject_eeg_full_info[idx][video_idx][0])
            target_label = 0 if target_label <= self.threshold else 1
            label_array[video_idx] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array