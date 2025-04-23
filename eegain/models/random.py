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
#from eegain.models import *
from config import *

# Random model that predicts classes based on their distribution in the data
class RandomModel_class_distribution:
    def __init__(self, train_dataloader, val_dataloader=None):
        # Collect labels from both train and validation data
        labels = []
        for x_batch, y_batch in train_dataloader:
            labels.extend(y_batch.tolist())
        
        # Add validation data if provided
        if val_dataloader is not None:
            for x_batch, y_batch in val_dataloader:
                labels.extend(y_batch.tolist())
                
        labels_f = [int(x) for x in labels]
        uniques = list(set(labels_f))
        print("this is uniques : ", uniques)
        self.num_classes = len(uniques)
        
        # Calculate weights based on class distribution
        weights = []
        for i in range(0, len(uniques)):
            portion = labels_f.count(i) / len(labels_f)
            weights.append(portion)
        self.weights = weights
        print(f"weights: {self.weights} (from {len(labels)} total samples)")

    def __call__(self, x):
        class_list = list(range(self.num_classes))
        return torch.tensor(np.random.choice(class_list, size=x.shape[0], p=self.weights))

# Random model that always predicts the most occuring class
class RandomModel_most_occurring:
    def __init__(self, train_dataloader, val_dataloader=None):
        self.majority_class = self.calculate_majority_class(train_dataloader, val_dataloader)

    def calculate_majority_class(self, train_dataloader, val_dataloader=None):
        labels = []
        # Collect labels from training data
        for x_batch, y_batch in train_dataloader:
            labels.extend(y_batch.tolist())
        
        # Collect labels from validation data if provided
        if val_dataloader is not None:
            for x_batch, y_batch in val_dataloader:
                labels.extend(y_batch.tolist())
                
        majority_class = max(set(labels), key=labels.count)
        print(f"Most occurring class: {majority_class} (from {len(labels)} total samples)")
        return majority_class

    def __call__(self, x):
        return torch.tensor([self.majority_class] * x.shape[0])