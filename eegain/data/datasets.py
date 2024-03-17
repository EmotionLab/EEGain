import os
import re
import mne
import torch
import pickle
import logging
import scipy.io
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from pathlib import Path
from scipy.io import loadmat
from ..transforms import Construct
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple
from torch.utils.data import Dataset

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
    def _create_user_recording_mapping(data_path: Path) -> Dict[int, List[str]]:
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

        logger.debug(f"Subject id -> sessions: {user_session_info}")
        return user_session_info

    def __init__(self, root: str, label_type: str, ground_truth_threshold, transform: Construct = None, **kwargs):
        """This is just constructor for SEED IV class
        Args:
            root(str): Path to SEED IV dataset folder
            label_type(str): 'V' for Valence and 'A' for Arousal
            transform(Construct): transformations to apply on dataset

        Return:
        """

        self.root = Path(root)
        self.ground_truth_threshold = ground_truth_threshold
        self.transform = transform
        self.mapping_list = SeedIV._create_user_recording_mapping(self.root)
        self.label_type = label_type
        self._num_trials = 24
        self._sampling_rate = 1000

        logger.info(f"Using Dataset: {self.__class__.__name__}")
        logger.info(f"Using label: {self.label_type}")

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

        logger.debug(
            f"subj index: {subject_index} data {data_array.keys()}, label {label_array.keys()}"
        )

        return data_array, label_array

    def __get_trials__(self, sessions, subject_ids):
        labels_file_path = self.root / Path("ReadMe.txt")
        path_to_channels_excel = self.root / Path("Channel Order.xlsx")
        channels_file = pd.read_excel(path_to_channels_excel, header=None)
        channels = list(channels_file.iloc[:, 0])

        num_trials = self._num_trials
        sampling_rate = self._sampling_rate
        data_array, label_array = {}, {}

        for session in sessions:
            path_to_mat = self.root / Path("eeg_raw_data") / Path(str(session))  # eeg_raw_data
            mat_data = scipy.io.loadmat(path_to_mat)  # Get Matlab File
            mat_data_values = list(mat_data.values())[
                              3:]  # Matlab file contains some not necessary info so let's remove it
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
                session_labels = eval(session_labels[session_labels.index("["):])

                emotional_label = session_labels[trial - 1]
                label_array[session_trial] = emotional_label

        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array


class MAHNOB(EEGDataset):
    @staticmethod
    def _create_user_recording_mapping(data_path: Path) -> Dict[int, List[str]]:
        """This method creates mapping between users and user_recordings
        Args:
            data_path(Path): path to MAHNOB-HCI dataset

        Returns:
            user_session_info(Dict[int, List[str]]): This is the dictionary where key is user_id and value is list
                                                     of file_names that's associated to this particular user
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

    def __init__(self, root: str, label_type: str, ground_truth_threshold, transform: Construct = None, **kwargs):
        """This is just constructor for MAHNOB class
        Args:
            root(str): Path to MAHNOB HCI dataset folder
            label_type(str): 'V' for Valence and 'A' for Arousal
            transform(Construct): transformations to apply on dataset

        Return:
        """
        self.root = Path(root)
        self.ground_truth_threshold = ground_truth_threshold
        self.transform = transform
        self.mapping_list = MAHNOB._create_user_recording_mapping(self.root)
        self.label_type = label_type

        logger.info(f"Using Dataset: {self.__class__.__name__}")
        logger.info(f"Using label: {self.label_type}")

    def __get_subject_ids__(self) -> List[int]:
        """Method returns list of subject ids"""
        return list(self.mapping_list.keys())

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
            target_label = 0 if target_label <= self.ground_truth_threshold else 1
            label_array[session_id] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        logger.debug(
            f"subj index: {subject_index} data {data_array.keys()}, label {label_array.keys()}"
        )
        return data_array, label_array

    def __get_trials__(self, session_ids: List, subject_video_mapping=None):
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
            target_label = 0 if target_label <= self.ground_truth_threshold else 1
            label_array[session_id] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array


class DEAP(EEGDataset):
    @staticmethod
    def _create_user_recording_mapping(data_path: Path) -> Dict[str, List[str]]:
        """This method creates mapping between users and user_recordings
        Args:
            data_path(Path): path to DEAP dataset

        Returns:
            user_session_info(Dict[str, List[str]]): This is the dictionary where key is user_id and value is list
                                                     of file_names that's associated to this particular user
        """
        file_paths = os.listdir(data_path)
        user_session_info = {}
        for subj in file_paths:
            if ".dat" in subj:
                id = subj[1:-4]
                # user_session_info[id] = [i for i in range(40)] this is the same as below, just this is hardcoded
                with open(os.path.join(data_path, subj), "rb") as file:
                    data = pickle.load(file, encoding="latin-1")
                    num_sessions = int(data['labels'].shape[0])
                    user_session_info[id] = [i for i in range(num_sessions)]
                    # if transform:
                    #     # TODO - this should be done with transform itself but transform doesn't work with
                    #     # this data because it's read by pickle and type of data['data'] is numpy ndarray
                    #     data['data'] = data['data'][:, :32, :]

        logger.debug(f"Subject id -> sessions: {user_session_info}")
        return user_session_info

    def __init__(
        self,
        root: str,
        label_type: str,
        ground_truth_threshold,
        transform: Construct = None,
        **kwargs
    ):
        self.root = Path(root)
        self.ground_truth_threshold = ground_truth_threshold
        self.transform = transform
        self.label_type = label_type
        self.mapping_list = self._create_user_recording_mapping(self.root)
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

        logger.info(f"Using Dataset: {self.__class__.__name__}")
        logger.info(f"Using label: {self.label_type}")

    def __get_subject_ids__(self) -> List[int]:
        """Method returns list of subject ids"""
        return self.mapping_list.keys()

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
        subject_index = str(subject_index) if subject_index>9 else "0"+str(subject_index)
        path_to_subject = (
                self.root / f's{subject_index}.dat'
        )
        with open(path_to_subject, "rb") as file:
            subject_data = pickle.load(file, encoding="latin-1")
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
            target_label = 0 if target_label <= self.ground_truth_threshold else 1
            label_array[session_id] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        logger.debug(
            f"subj index: {subject_index} data {data_array.keys()}, label {label_array.keys()}"
        )
        return data_array, label_array

    def __get_trials__(self, session_ids: List, subject_id: str):
        data_array, label_array = {}, {}
        path_to_subject = (
                self.root / f's{subject_id}.dat'
        )
        with open(path_to_subject, "rb") as file:
            subject_data = pickle.load(file, encoding="latin-1")
        for session_id in session_ids:
            curr_labels = subject_data['labels'][session_id]
            curr_data = subject_data['data'][session_id]

            curr_data = mne.io.RawArray(curr_data, self.info, verbose=False)

            if self.transform:
                curr_data = self.transform(curr_data)

            felt_valence = curr_labels[0]
            felt_arousal = curr_labels[1]

            data_array[session_id] = curr_data
            label_array[session_id] = np.array([felt_arousal, felt_valence])

        # choose label and split into binary
        idx = 0 if self.label_type == "A" else 1
        for session_id, data in label_array.items():
            target_label = data[idx]
            target_label = 0 if target_label <= self.ground_truth_threshold else 1
            label_array[session_id] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array


class DREAMER(EEGDataset):
    @staticmethod
    def _create_user_recording_mapping(data_path: Path) -> Dict[int, List[str]]:
        """This method creates mapping between users and user_recordings
        Args:
            data_path(Path): path to DREAMER dataset

        Returns:
            user_session_info(Dict[int, List[str]]): This is the dictionary where key is user_id and value is list
                                                     of file_names that's associated to this particular user
        """
        user_session_info = {}

        mat_data = scipy.io.loadmat(data_path)

        num_subjects = len(list(mat_data["DREAMER"][0][0][0][0]))
        for curr_subject_idx in range(num_subjects):
            num_videos_curr_subject = len(list(mat_data["DREAMER"][0][0][0][0][curr_subject_idx][0][0][2][0][0][1]))
            video_idxs = [i for i in range(num_videos_curr_subject)]
            user_session_info[curr_subject_idx] = video_idxs

        logger.debug(f"Subject id -> sessions: {user_session_info}")
        return user_session_info

    def __init__(self, root: str, label_type: str, ground_truth_threshold, transform: Construct = None, **kwargs):
        """This is just constructor for DREAMER class
        Args:
            root(str): Path to DREAMER.mat file
            label_type(str): 'V' for Valence and 'A' for Arousal
            transform(Construct): transformations to apply on dataset

        Return:
        """
        self.root = Path(root)
        self.ground_truth_threshold = ground_truth_threshold
        self.transform = transform
        self.label_type = label_type
        self.num_videos = 18
        self.threshold = 3
        self.mapping_list = self._create_user_recording_mapping(
            self.root
            # , self.transform, self.label_type
        )

        logger.info(f"Using Dataset: {self.__class__.__name__}")
        logger.info(f"Using label: {self.label_type}")

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
            raw_data = mne.io.RawArray(person_eeg_recording, info, verbose=False)

            if self.transform:
                raw_data = self.transform(raw_data)

            data_array[video_idx] = raw_data.get_data()

            target_label = int(subject_eeg_full_info[idx][video_idx][0])
            target_label = 0 if target_label <= self.ground_truth_threshold else 1
            label_array[video_idx] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        logger.debug(
            f"subj index: {subject_index} data {data_array.keys()}, label {label_array.keys()}"
        )

        return data_array, label_array

    def __get_trials__(self, video_ids, subject_id):
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
            raw_data = mne.io.RawArray(person_eeg_recording, info, verbose=False)

            if self.transform:
                raw_data = self.transform(raw_data)

            data_array[video_idx] = raw_data.get_data()

            target_label = int(subject_eeg_full_info[idx][video_idx][0])
            target_label = 0 if target_label <= self.ground_truth_threshold else 1
            label_array[video_idx] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array


class AMIGOS(EEGDataset):
    @staticmethod
    def _create_user_csv_mapping(
        file_paths: List[str], data_path: Path, transform, preprocessed: bool = True
    ) -> Dict[int, List[str]]:
        """This method creates mapping between users and user_recordings
        Args:
            data_path(Path): path to AMIGOS dataset

        Returns:
            user_session_info(Dict[int, List[str]]): This is the dictionary where key is user_id and value is list
                                                     of file_names that's associated to this particular user
        """
        user_session_info = {}

        for curr_path in file_paths:
            if "Preprocessed" in curr_path or "Original" in curr_path:
                curr_full_path = data_path / Path(curr_path) / Path(f"{curr_path}.mat")
                id = curr_path.split("/")[-1].split("_")[-1][1:]
                data = loadmat(curr_full_path)
                _data = {"data": data["joined_data"][0], "labels": data["labels_selfassessment"][0]}
                num_videos = len(_data['data'])
                user_session_info[id] = [i for i in range(num_videos)]  # _data

        logger.debug(f"Subject id -> sessions: {user_session_info}")
        return user_session_info

    def __init__(
        self,
        root: str,
        label_type: str,
        ground_truth_threshold,
        preprocessed: bool = True,
        transform: Construct = None,
        **kwargs
    ):
        root_path = Path(root) / Path("Physiological Recordings") / Path("Matlab Preprocessed Data") if preprocessed \
            else Path(root) / Path("Physiological Recordings") / Path("Matlab Original Data")
        self.root = root_path
        self.preprocessed = preprocessed
        self.ground_truth_threshold = ground_truth_threshold
        self.transform = transform
        self.file_paths = os.listdir(self.root)
        self.label_type = label_type
        self.mapping_list = self._create_user_csv_mapping(
            self.file_paths, self.root, self.transform, preprocessed
        )
        self._ch_names = [
            "AF3",
            "F7",
            "F3",
            "FC5",
            "T7",
            "P7",
            "O1",
            "O2",
            "P8",
            "T8",
            "FC6",
            "F4",
            "F8",
            "AF4",
            "ECG_Right",
            "ECG_Left",
            "GSR"
        ]

        self._sfreq = 128
        self.info = mne.create_info(
            self._ch_names, self._sfreq, ch_types="misc", verbose=None
        )

        logger.info(f"Using Dataset: {self.__class__.__name__}")
        logger.info(f"Using label: {self.label_type}")

    def __get_subject_ids__(self) -> List[int]:
        """Method returns list of subject ids"""
        return self.mapping_list.keys()

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
        if self.preprocessed:
            subject_data = loadmat(os.path.join(self.root, f"Data_Preprocessed_P{subject_index}/Data_Preprocessed_P{subject_index}.mat"))
        else:
            subject_data = loadmat(os.path.join(self.root, f"Data_Original_P{subject_index}/Data_Original_P{subject_index}.mat"))
        # TODO - data original doesn't have "joined_data
        subject_data = {"data": subject_data["joined_data"][0], "labels": subject_data["labels_selfassessment"][0]}
        # subject_data = self.mapping_list[subject_index]
        datas = subject_data["data"]
        labels = subject_data["labels"]
        data_array, label_array = {}, {}
        shp = 16 if datas.shape[0] > 16 else datas.shape[0] # this is because remaining videos are long ones and not everyone have that
        for i in range(0, shp):
            data = datas[i]
            data = mne.io.RawArray(data.T, self.info, verbose=False)

            if self.transform:
                data = self.transform(data)
            data_array[i] = data.get_data()
            felt_arousal = float(labels[i][0][0])
            felt_valence = float(labels[i][0][1])
            label_array[i] = np.array([felt_arousal, felt_valence])

        # choose label and split into binary
        idx = 0 if self.label_type == "A" else 1
        for session_id, data in label_array.items():
            target_label = data[idx]
            target_label = 0 if target_label <= self.ground_truth_threshold else 1
            label_array[session_id] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        logger.debug(
            f"subj index: {subject_index} data {data_array.keys()}, label {label_array.keys()}"
        )

        return data_array, label_array

    def __get_trials__(self, session_ids: List, subject_id: Dict):
        data_array, label_array = {}, {}
        for session_id in session_ids:
            if self.preprocessed:
                data = loadmat(os.path.join(self.root, f"Data_Preprocessed_P{subject_id}/Data_Preprocessed_P{subject_id}.mat"))
            else:
                data = loadmat(os.path.join(self.root, f"Data_Original_P{subject_id}/Data_Original_P{subject_id}.mat"))
            # TODO - data original doesn't have "joined_data
            _data = {"data": data["joined_data"][0], "labels": data["labels_selfassessment"][0]}

            curr_labels = _data['labels'][session_id][0]
            curr_data = _data['data'][session_id]

            felt_valence = curr_labels[1]
            felt_arousal = curr_labels[0]

            curr_data = curr_data.T
            curr_data = mne.io.RawArray(
                curr_data, self.info, verbose=False
            )  # convert numpy ndarray to mne object

            if self.transform:
                curr_data = self.transform(curr_data)

            data_array[session_id] = curr_data
            label_array[session_id] = np.array([felt_arousal, felt_valence])

        # choose label and split into binary
        idx = 0 if self.label_type == "A" else 1
        for session_id, data in label_array.items():
            target_label = data[idx]
            target_label = 0 if target_label <= self.ground_truth_threshold else 1
            label_array[session_id] = target_label

        # expand dims
        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array


class Seed(EEGDataset):
    @staticmethod
    def _create_user_mat_mapping(data_path: Path) -> Dict[int, List[str]]:
        """This method creates mapping between users and user_recordings
        Args:
            data_path(Path): path to SEED dataset

        Returns:
            user_session_info(Dict[int, List[str]]): This is the dictionary where key is user_id and value is list
                                                     of file_names that's associated to this particular user
        """

        num_sessions = 3  # There are three sessions in SEED IV dataset
        user_session_info: Dict[int, List[str]] = defaultdict(list)
        path = (
                data_path / Path("SEED_EEG/Preprocessed_EEG")  # eeg_preprocessed_data
        )
        file_paths = os.listdir(path)
        for mat_file_name in file_paths:
            if "label" not in mat_file_name and "channel" not in mat_file_name:
                subject_id = int(
                    mat_file_name[: mat_file_name.index("_")]
                )  # file name starts with user_id
                user_session_info[subject_id].append(
                    str(path) + "/" + mat_file_name
                )

        return user_session_info

    def __init__(self, root: str, label_type: str, transform: Construct = None, **kwargs):
        """This is just constructor for SEED class
        Args:
            root(str): Path to SEED dataset folder
            transform(Construct): transformations to apply on dataset

        Return:
        """

        self.root = Path(root)
        self.transform = transform
        self.mapping_list = Seed._create_user_mat_mapping(self.root)
        self.label_type = label_type
        self._num_trials = 15
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

        path_to_channels_excel = self.root / Path("SEED_EEG/channel-order.xlsx")
        channels_file = pd.read_excel(path_to_channels_excel, header=None)
        channels = list(channels_file.iloc[:, 0])

        num_trials = self._num_trials
        sampling_rate = self._sampling_rate
        sessions = self.mapping_list[subject_index]
        data_array, label_array = {}, {}

        for session in sessions:
            path_to_mat = self.root / Path("Preprocessed_EEG") / Path(str(session)) # eeg_raw_data
            mat_data = scipy.io.loadmat(path_to_mat)  # Get Matlab File
            mat_data_values = list(mat_data.values())[3:]  # Matlab file contains some not necessary info so let's remove it
            for trial in range(1, num_trials + 1):
                eeg_data = mat_data_values[
                    trial - 1
                ]  # each matlab file contains 15 trials. Here we take one trial
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

                # Extract label from file and add to label_array
                session_id = session[: session.index("/")]
                session_labels = [ 2,  1, 0, 0,  1,  2, 0,  1,  2,  2,  1, 0,  1,  2, 0]

                emotional_label = session_labels[trial - 1]
                label_array[session_trial] = emotional_label

        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array

    def __get_trials__(self, sessions, subject_ids):
        path_to_channels_excel = self.root / Path("SEED_EEG/channel-order.xlsx")
        channels_file = pd.read_excel(path_to_channels_excel, header=None)
        channels = list(channels_file.iloc[:, 0])

        num_trials = self._num_trials
        sampling_rate = self._sampling_rate
        data_array, label_array = {}, {}

        for session in sessions:
            path_to_mat = self.root / Path("SEED_EEG") / Path(str(session))  # eeg_raw_data
            mat_data = scipy.io.loadmat(path_to_mat)  # Get Matlab File
            mat_data_values = list(mat_data.values())[
                              3:]  # Matlab file contains some not necessary info so let's remove it
            for trial in range(1, num_trials + 1):
                eeg_data = mat_data_values[
                    trial - 1
                    ]  # each matlab file contains 15 trials. Here we take one trial
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
                session_id = session[: session.index("/")]
                # session_labels = [ 1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1]
                session_labels = [ 2,  1, 0, 0,  1,  2, 0,  1,  2,  2,  1, 0,  1,  2, 0]
                emotional_label = session_labels[trial - 1]
                label_array[session_trial] = emotional_label

        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array
