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

# Random model that always predicts the most occuring class in the dataset
# class RandomModel:
#     def __init__(self, train_dataloader):
#         labels = []
#         for x_batch, y_batch in train_dataloader:
#             labels.extend(y_batch)
#         labels_f = [int(x) for x in labels]
#         uniques = list(set(labels_f))
#         print("this is uniques : ", uniques)
#         self.num_classes = len(uniques)
#         self.most_occurring_class = max(set(labels_f), key=labels_f.count)
#         print("Most occurring class:", self.most_occurring_class)

#     def __call__(self, x):
#         return torch.tensor([self.most_occurring_class] * x.shape[0])
class RandomModel:
    def __init__(self, train_dataloader):
        self.majority_class = self.calculate_majority_class(train_dataloader)

    def calculate_majority_class(self, train_dataloader):
        labels = []
        for x_batch, y_batch in train_dataloader:
            labels.extend(y_batch.tolist())
        majority_class = max(set(labels), key=labels.count)
        print("Most occurring class:", majority_class)
        return majority_class

    def __call__(self, x):
        return torch.tensor([self.majority_class] * x.shape[0])

## Test code for random model for LOTO maybe?
"""This code calculates the class frequencies for each label across all sessions, 
    and then uses a weighted average to determine the most occurring class. """
# class RandomModel:
#     def __init__(self, train_dataloader):
#         self.class_frequencies = {}
#         for x_batch, y_batch in train_dataloader:
#             labels = y_batch.tolist()
#             for label in labels:
#                 if label not in self.class_frequencies:
#                     self.class_frequencies[label] = 0
#                 self.class_frequencies[label] += 1
#         self.total_labels = sum(self.class_frequencies.values())
#
#         self.weighted_class_frequencies = {label: freq / self.total_labels for label, freq in self.class_frequencies.items()}
#         self.most_occurring_class = max(self.weighted_class_frequencies, key=self.weighted_class_frequencies.get)
#         print("Most occurring class:", self.most_occurring_class)

#     def __call__(self, x):
#         return torch.tensor([self.most_occurring_class] * x.shape[0])