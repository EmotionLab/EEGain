import os
import click
from dataclasses import asdict
# os.environ["LOG_LEVEL"] = "DEBUG"

import torch
from torch import nn
from tqdm import tqdm

import eegain
from eegain.data import EEGDataloader
from eegain.data.datasets import *
from eegain.logger import EmotionLogger
from eegain.models import *
from config import *


class RandomModel:
    def __init__(self, train_dataloader):
        labels = []
        for x_batch, y_batch in train_dataloader:
            labels.extend(y_batch)

        portion_0 = labels.count(0) / len(labels)
        portion_1 = labels.count(1) / len(labels)

        self.weights = [portion_0, portion_1]
        print("weights", self.weights)

    def __call__(self, x):
        return torch.tensor(np.random.choice([0, 1], size=x.shape[0], p=self.weights))