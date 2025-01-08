import os
import click
from dataclasses import asdict
# os.environ["LOG_LEVEL"] = "DEBUG"

import torch
from torch import nn
from tqdm import tqdm

import eegain
from eegain.data import EEGDataloader
from eegain.data.datasets_org import *
from eegain.logger import EmotionLogger
from eegain.models import *
from config import *


# class RandomModel:
#     def __init__(self, train_dataloader):
#         labels = []
#         for x_batch, y_batch in train_dataloader:
#             labels.extend(y_batch)
#         labels_f = [int(x) for x in labels]
#         uniques = list(set(labels_f))
#         print("this is uniques : ", uniques)
#         self.num_classes = len(uniques)
#         weights = []
#         for i in range(0,len(uniques)):
#             portion_0 = labels.count(i) / len(labels)
#             weights.append(portion_0)
#         self.weights = weights
#         print("weights", self.weights)

#     def __call__(self, x):
#         class_list = list(range(self.num_classes))
#         return torch.tensor(np.random.choice(class_list, size=x.shape[0], p=self.weights))

class RandomModel:
    def __init__(self, train_dataloader):
        labels = []
        for x_batch, y_batch in train_dataloader:
            labels.extend(y_batch)
        labels_f = [int(x) for x in labels]
        uniques = list(set(labels_f))
        print("this is uniques : ", uniques)
        self.num_classes = len(uniques)
        self.most_occurring_class = max(set(labels_f), key=labels_f.count)
        print("Most occurring class:", self.most_occurring_class)

    def __call__(self, x):
        return torch.tensor([self.most_occurring_class] * x.shape[0])