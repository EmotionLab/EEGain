import os

os.environ["LOG_LEVEL"] = "DEBUG"

import torch
from torch import nn
from tqdm import tqdm
import config
import eegain
from eegain.data import EEGDataloader
from eegain.data.datasets import DEAP, MAHNOB, SeedIV
from eegain.logger import EmotionLogger
from eegain.models import DeepConvNet, EEGNet, ShallowConvNet, TSception
from collections import defaultdict
from sklearn.metrics import *
from helpers import main_loso, main_loto

# -------------- Preprocessing --------------
transform = eegain.transforms.Construct(
    [
        eegain.transforms.Crop(t_min=30, t_max=-30),
        eegain.transforms.DropChannels(
            [
                "EXG1",
                "EXG2",
                "EXG3",
                "EXG4",
                "EXG5",
                "EXG6",
                "EXG7",
                "EXG8",
                "GSR1",
                "GSR2",
                "Erg1",
                "Erg2",
                "Resp",
                "Temp",
                "Status",
            ]
        ),
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(s_rate=128),
        eegain.transforms.Segment(duration=4, overlap=0),
    ]
)


# -------------- Dataset --------------
mahnob_dataset = MAHNOB(
    root=config.DataConfig.data_path,
    label_type=config.DataConfig.label_type,
    ground_truth_threshold=config.DataConfig.ground_truth_threshold,
    transform=transform,
)

# -------------- Model --------------
model = TSception(
                num_classes=config.DataConfig.n_classes,
                input_size=config.TScepctionConfig.input_size,
                sampling_r=config.TScepctionConfig.sampling_r,
                num_t=config.TScepctionConfig.num_t,
                num_s=config.TScepctionConfig.num_s,
                hidden=config.TScepctionConfig.hidden,
                dropout_rate=config.TScepctionConfig.dropout_rate,
            )


main_loto(mahnob_dataset, model)
