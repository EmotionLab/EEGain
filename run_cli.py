import os
import copy

os.environ["LOG_LEVEL"] = "DEBUG"

import torch
from torch import nn
from tqdm import tqdm
import config
import click
import eegain
from eegain.data import EEGDataloader
from eegain.data.datasets import DEAP, MAHNOB, SeedIV, AMIGOS, DREAMER, Seed
from eegain.logger import EmotionLogger
from eegain.models import DeepConvNet, EEGNet, ShallowConvNet, TSception
from collections import defaultdict
from dataclasses import asdict

from sklearn.metrics import *
from helpers import main_loso, main_loto, main_loso_fixed, setup_seed
from config import *
from colorama import Fore, Style
import functools

MAHNOB_transform = [
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
                    #"Oz", "Pz", "Fz", "Cz" # remove Oz, Pz, Fz, Cz channels to replicate the TSception paper
                ]
            ),
            eegain.transforms.Filter(l_freq=0.3, h_freq=45),
            eegain.transforms.NotchFilter(freq=50),
            eegain.transforms.Resample(sampling_r=128),
        ]

DEAP_transform = [
        eegain.transforms.Crop(t_min=3, t_max=None),
        eegain.transforms.DropChannels(
            [
                "EXG1",
                "EXG2",
                "EXG3",
                "EXG4",
                "GSR1",
                "Plet",
                "Resp",
                "Temp",
                #"Oz", "Pz", "Fz", "Cz" # remove Oz, Pz, Fz, Cz channels to replicate the TSception paper
            ]
        ),
        eegain.transforms.NotchFilter(freq=50), # comment this line to replicate the TSception paper
        eegain.transforms.Resample(sampling_r=128)
    ]

AMIGOS_transform = [
        eegain.transforms.DropChannels(
            [
            "ECG_Right",
            "ECG_Left",
            "GSR"
            ]
        ),
        eegain.transforms.NotchFilter(freq=50),
    ]

DREAMER_transform = [
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(sampling_r=128),
    ]

SeedIV_transform = [
        # eegain.transforms.DropChannels(channels_to_drop_seed_iv),
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(sampling_r=128),
    ]

Seed_transform =  [
        # eegain.transforms.DropChannels(channels_to_drop_seed),
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(sampling_r=128),
    ]

def generate_options():
    def decorator(func):
        # First add required common options (like data_config)
        func = click.option("--data_config", default="MAHNOBConfig", required=False, type=str, 
                           help="Dataset config class name (e.g., DEAPConfig)")(func)
        
        # Get all dataset config classes
        dataset_configs = [DEAPConfig, MAHNOBConfig, AMIGOSConfig, DREAMERConfig, SeedIVConfig, SeedConfig]
        
        # Get all other config classes
        other_configs = [TransformConfig, TrainingConfig, EEGNetConfig, TSceptionConfig, DeepConvNetConfig, ShallowConvNetConfig]
        
        # Generate CLI options for all config values from all configs
        all_configs = other_configs + dataset_configs
        
        # Track all fields we've seen to handle duplicates
        seen_fields = set()
        
        # Add options for all config fields (both dataset and non-dataset)
        for config_class in all_configs:
            config_instance = config_class()
            for field, value in asdict(config_instance).items():
                if field not in seen_fields:
                    seen_fields.add(field)
                    option = click.option(f"--{field}", default=value, required=False, type=type(value))
                    func = option(func)
        
        @functools.wraps(func)
        def wrapper(**kwargs):
            # Get the specified dataset config
            config_class_name = kwargs.get("data_config")
            config_class = globals().get(config_class_name)
            
            if config_class is None:
                raise ValueError(f"Config class {config_class_name} not found")
            
            # Apply defaults from the selected dataset config
            config_instance = config_class()
            for field, value in asdict(config_instance).items():
                # Update all values from the specific config class, unless explicitly provided in CLI
                if not click.get_current_context().get_parameter_source(field) == click.core.ParameterSource.COMMANDLINE:
                    kwargs[field] = value
            
            return func(**kwargs)
        
        return wrapper
    return decorator

@click.command()
@click.option("--model_name", required=True, type=str, help="name of the model")
@click.option("--data_name", required=True, type=str, help="name of the dataset")
# new options for logging predictions
@click.option("--log_predictions", type=bool, help="log predictions to a directory")
@click.option("--log_predictions_dir", type=str, help="directory to save logged predictions")
@click.option("--train_val_split", type=float, default=0.8, help="ratio of training data to use for training (rest for validation)")
@click.option("--random_seed", type=int, default=2025, help="random seed for reproducibility")
@generate_options()

def main(**kwargs):
    setup_seed(kwargs["random_seed"])
    print(f"[INFO] Using random seed: {kwargs['random_seed']}")
    # -------------- Data --------------
    transform = globals().get(kwargs["data_name"] + "_transform")
    # Update sampling_r in any Resample transform to match CLI parameter
    for i, t in enumerate(transform):
        if isinstance(t, eegain.transforms.Resample):
            if int(t.sampling_r) != int(kwargs['sampling_r']):
                print(f"[INFO] Updating sampling rate from {t.sampling_r} to {kwargs['sampling_r']}")
            # Create a new Resample transform with updated sampling_r
            transform[i] = eegain.transforms.Resample(sampling_r=kwargs["sampling_r"])
            
    transform.append(eegain.transforms.Segment(duration=kwargs["window"], overlap=kwargs["overlap"]))
    transform = eegain.transforms.Construct(transform)

    dataset = globals()[kwargs['data_name']](transform=transform, root=kwargs["data_path"], **kwargs)
    
    # [NEW] Log predictions if the flag is set to True and create the directory if it does not exist
    if kwargs["log_predictions"]:
        if not os.path.exists(kwargs["log_predictions_dir"]):
            os.makedirs(kwargs["log_predictions_dir"])
        print(f"[INFO] Logger: Logging predictions to directory: {kwargs['log_predictions_dir']}")

    # -------------- RANDOM Model --------------
    if kwargs["model_name"]=="RANDOM_most_occurring":
        print("initializing random model with most occurring class")
        print(Fore.RED + "[NOTE] You have selected Random Model that always predicts the most occurring class in the training and validation sets, so it is not recommended to use it for F1-score calculations.")
        print(Fore.RED + "For F1-score calculations, please use the Random Model that predicts a random class based on class distribution (RANDOM_class_distribution)"+ Style.RESET_ALL)
        model = None
        empty_model = None
    elif kwargs["model_name"]=="RANDOM_class_distribution":
        print("initializing random model with class distribution")
        print(Fore.RED + "[NOTE] You have selected Random Model that predicts a random class based on class distribution in the training and validation sets, so it is not recommended to use it for Accuracy calculations.")
        print(Fore.RED + "For Accuracy calculations, please use the Random Model that always predicts the most occurring class (RANDOM_most_occurring)"+ Style.RESET_ALL)
        model = None
        empty_model = None
    # -------------- Model --------------
    else:
        model = globals()[kwargs['model_name']](input_size=[1, kwargs["channels"], kwargs["window"]*kwargs["sampling_r"]], **kwargs)
        empty_model = copy.deepcopy(model)
        
    if kwargs["split_type"] == "LOSO":
        classes = [i for i in range(kwargs["num_classes"])]
        main_loso(dataset, model, empty_model, classes, **kwargs)
    elif kwargs["split_type"] == "LOSO_Fixed":
        classes = [i for i in range(kwargs["num_classes"])]
        main_loso_fixed(dataset, model, empty_model, classes, **kwargs)
    else:
        classes = [i for i in range(kwargs["num_classes"])]
        main_loto(dataset, model, empty_model, classes, **kwargs)

if __name__ == "__main__":
    main()
