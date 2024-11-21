import os
import csv
os.environ["LOG_LEVEL"] = "DEBUG"

import torch
from torch import nn
from tqdm import tqdm

import copy
import config
import eegain
import json
from eegain.data import EEGDataloader
from eegain.data.datasets import DEAP, MAHNOB, SeedIV
from eegain.logger import EmotionLogger
from eegain.models import DeepConvNet, EEGNet, ShallowConvNet, TSception, RandomModel
from collections import defaultdict
from sklearn.metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_one_epoch(model, loader, loss_fn):
    model.eval()
    all_preds, all_actuals = [], []
    dataset_size, running_loss, epoch_loss = 0, 0, None

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Test ")
    with torch.no_grad():
        for i, (x_batch, y_batch) in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            out = model(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)

            batch_size = x_batch.size(0)
            dataset_size += batch_size
            all_preds.extend(pred.data.tolist())
            all_actuals.extend(y_batch.data.tolist())
            running_loss += loss.item() * batch_size
            epoch_loss = running_loss / dataset_size
            pbar.set_postfix(loss=f"{epoch_loss:0.4f}")
            # if i % 10 == 0:  # Print every 10 batches
            #     print(f"Batch {i} - Pred: {pred}, Actual: {y_batch}")
            print(f"TEST Batch {i}: Size {batch_size}, Pred {len(pred.tolist())}, Actual {len(y_batch.tolist())}")
    return all_preds, all_actuals, epoch_loss

def test_one_epoch_random(model, loader, loss_fn):
    all_preds, all_actuals = [], []
    dataset_size, running_loss, epoch_loss = 0, 0, None

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Test ")
    with torch.no_grad():
        for i, (x_batch, y_batch) in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)

            batch_size = x_batch.size(0)
            dataset_size += batch_size

            all_preds.extend(pred.data.tolist())
            all_actuals.extend(y_batch.data.tolist())
            epoch_loss = 0
            pbar.set_postfix(loss=f"{epoch_loss:0.4f}")

    return all_preds, all_actuals, epoch_loss


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    all_preds, all_actuals = [], []
    dataset_size, running_loss, epoch_loss = 0, 0, None

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train ")
    for i, (x_batch, y_batch) in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        out = model(x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        dataset_size += batch_size
        all_preds.extend(pred.data.tolist())
        all_actuals.extend(y_batch.data.tolist())
        running_loss += loss.item() * batch_size
        epoch_loss = running_loss / dataset_size
        pbar.set_postfix(loss=f"{epoch_loss:0.4f}")
        # if i % 10 == 0:  # Print every 10 batches
        #     print(f"Batch {i} - Pred: {pred}, Actual: {y_batch}")
        #print(f"Train Batch {i}: Size {batch_size}, Pred {len(pred.tolist())}, Actual {len(y_batch.tolist())}")
    return all_preds, all_actuals, epoch_loss


def run_loto(
        model,
        train_dataloader,
        test_dataloader,
        test_ids,
        optimizer,
        scheduler,
        loss_fn,
        epoch,
        logger,
        random_baseline,
        subject_id,     # new params
        split_type="LOTO",
        **kwargs       # new params
):
    ## [NEW CODE BLOCK]
    # if you want to log predictions, this code block will create the directory and file
    if kwargs["log_predictions"] == True:
        # Ensure the directory exists
        prediction_dir = kwargs["log_predictions_dir"]
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
    
        prediction_file = os.path.join(prediction_dir, f"predictions_loto_{test_ids[0]}.csv")
        with open(prediction_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Data Part', 'Video id', 'Subject id', 'Actual', 'Predicted'])
    
    for i in range(epoch):
        print(f"\nEpoch {i}/{epoch}")
        if not random_baseline:
            train_pred, train_actual, train_loss = train_one_epoch(
                model, train_dataloader, optimizer, loss_fn
            )
            test_pred, test_actual, test_loss = test_one_epoch(
            model, test_dataloader, loss_fn
            )
            scheduler.step()
        else:
            test_pred, test_actual, test_loss = test_one_epoch_random(
            model, test_dataloader, loss_fn
            )
            train_pred, train_actual, train_loss = test_pred, test_actual, test_loss
        if split_type != "LOTO":
            # in the end of epoch it logs metrics for this specific epoch. test_ids is test_session_ids
            logger.log(test_ids[0], train_pred, train_actual, i, "train", train_loss)
            logger.log(test_ids[0], test_pred, test_actual, i, "val", test_loss)
    
        ## [NEW CODE BLOCK]
        # if you want to log predictions, this code block will write the predictions to the file
        if kwargs["log_predictions"] == True:
            with open(prediction_file, 'a', newline='') as f:
                writer = csv.writer(f)

                for act, pred, subject_id in zip(test_actual, test_pred, subject_id):
                    writer.writerow([i, 'val', test_ids[0], subject_id, act, pred])
        
    if split_type == "LOTO":
        return train_pred, train_actual, test_pred, test_actual

    logger.log_summary()


def run_loso(
        model,
        train_dataloader,
        test_dataloader,
        test_subject_ids,
        optimizer,
        scheduler,
        loss_fn,
        epoch,
        logger,
        random_baseline,
        train_videos,   # new params
        test_videos,    # new params
        **kwargs        # new params
):
        
    print(f"test subject ids {test_subject_ids}")
    ## [NEW CODE BLOCK]
    # if you want to log predictions, this code block will create the directory and file
    if kwargs["log_predictions"] == True:
        # Ensure the directory exists
        prediction_dir = kwargs["log_predictions_dir"]
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
    
        prediction_file = os.path.join(prediction_dir, f"predictions_loso_{test_subject_ids[0]}.csv")
        with open(prediction_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Data Part', 'Subject id', 'Video id', 'Actual', 'Predicted'])

        prediction_file_log = os.path.join(prediction_dir, f"LOG_predictions_loso_{test_subject_ids[0]}.csv")
        with open(prediction_file_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Data Part', 'Subject id', 'Video id', 'Actual', 'Predicted'])
        
    for i in range(epoch):
        print(f"\nEpoch {i}/{epoch}")
        if not random_baseline:
            train_pred, train_actual, train_loss = train_one_epoch(
                model, train_dataloader, optimizer, loss_fn
            )
            
            test_pred, test_actual, test_loss = test_one_epoch(
                model, test_dataloader, loss_fn
            )
            print(f" len of test_pred is {len(test_pred)}")
            scheduler.step()
        else:
            test_pred, test_actual, test_loss = test_one_epoch_random(
                model, test_dataloader, loss_fn
            )
            train_pred, train_actual, train_loss = test_pred, test_actual, test_loss

        
        logger.log(test_subject_ids[0], train_pred, train_actual, i, "train", prediction_file_log, False, train_videos, train_loss)
        logger.log(test_subject_ids[0], test_pred, test_actual, i, "val", prediction_file_log, True, test_videos, test_loss)
        
        ## [NEW CODE BLOCK]
        # if you want to log predictions, this code block will write the predictions to the file
        if kwargs["log_predictions"] == True:
            with open(prediction_file, 'a', newline='') as f:
                writer = csv.writer(f)

                for act, pred, vid in zip(test_actual, test_pred, test_videos):
                    writer.writerow([i, 'val', test_subject_ids[0], vid, act, pred])

    logger.log_summary(overal_log_file="overal_log", log_dir="logs/")


def main_loto(dataset, model, empty_model, classes, **kwargs):
    subject_video_mapping = dataset.mapping_list
    logger = EmotionLogger(log_dir=kwargs["log_dir"], class_names=classes)

    for subject_id, session_ids in subject_video_mapping.items():
        n_fold=len(session_ids)
        n_fold=10
        eegloader = EEGDataloader(dataset, batch_size=32).loto(subject_id, session_ids,
                                                               n_fold=n_fold)  # pass n_fold=len(session_ids) for LOTO
        num_epoch = kwargs["num_epochs"]
        all_train_preds_for_subject, all_train_actuals_for_subject, all_test_preds_for_subject, all_test_actuals_for_subject = [], [], [], []
        for i, loader in enumerate(eegloader):
            if kwargs["model_name"]=="RANDOM":
                model = RandomModel(loader["train"])
                optimizer = None
                is_random=True
                scheduler=None
            else:
                model = copy.deepcopy(empty_model)
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

                is_random=False
            loss_fn = nn.CrossEntropyLoss(label_smoothing=kwargs["label_smoothing"])
            _, _, test_pred, test_actual = run_loto(
                model=model,
                train_dataloader=loader["train"],
                test_dataloader=loader["test"],
                test_ids=loader["test_session_indexes"],
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                epoch=num_epoch,
                logger=logger,
                random_baseline=is_random,
                subject_id=loader["subject_id"],    # new params
                split_type="LOTO",
                **kwargs,                           # new params   
            )
            all_test_preds_for_subject.append(test_pred)
            all_test_actuals_for_subject.append(test_actual)

        # all_model_state_dicts.append(model.state_dict())
        all_test_preds_for_subject = [item for sublist in all_test_preds_for_subject for item in sublist]
        all_test_actuals_for_subject = [item for sublist in all_test_actuals_for_subject for item in sublist]

        logger.log(subject_id, all_test_preds_for_subject, all_test_actuals_for_subject, num_epoch, "val")

    logger.log_summary(overal_log_file=kwargs["overal_log_file"], log_dir=kwargs["log_dir"])


def loso_loop(model, loader, logger, **kwargs):
    if kwargs["model_name"]=="RANDOM":
        optimizer = None
        is_random = True
        scheduler = None
    else:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"],
                                     weight_decay=kwargs["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

        is_random = False
    loss_fn = nn.CrossEntropyLoss(label_smoothing=kwargs["label_smoothing"])
    run_loso(
        model=model,
        train_dataloader=loader["train"],
        test_dataloader=loader["test"],
        test_subject_ids=loader["test_subject_indexes"],
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epoch=kwargs["num_epochs"],
        logger=logger,
        random_baseline=is_random,
        train_videos = loader["train_videos"],  # new params
        test_videos = loader["test_videos"],    # new params
        **kwargs
    )



def main_loso(dataset, model, empty_model, classes, **kwargs):
    eegloader = EEGDataloader(dataset, batch_size=32).loso()

    logger = EmotionLogger(log_dir=kwargs["log_dir"], class_names=classes)
    for loader in eegloader:
        if kwargs["model_name"]=="RANDOM":
            model = RandomModel(loader["train"])
        else:
            model = copy.deepcopy(empty_model)
        loso_loop(model, loader, logger, **kwargs)
    logger.log_summary(overal_log_file=kwargs["overal_log_file"], log_dir=kwargs["log_dir"])

def main_loso_fixed(dataset, model, empty_model, classes, **kwargs):
    dataset_name = dataset.__class__.__name__
    test_subjects_json_path = 'test_subjects.json'

    with open(test_subjects_json_path, 'r') as file:
        train_test_split_json = json.load(file)

    train_set = train_test_split_json[dataset_name]['train']
    test_set = train_test_split_json[dataset_name]['test']

    eegloader = EEGDataloader(dataset, batch_size=32).loso_fixed(train_set, test_set)

    logger = EmotionLogger(log_dir=kwargs["log_dir"], class_names=classes)
    if kwargs["model_name"]=="RANDOM":
        model = RandomModel(eegloader["train"])
    loso_loop(model, eegloader, logger, **kwargs)
    logger.log_summary(overal_log_file=kwargs["overal_log_file"], log_dir=kwargs["log_dir"])