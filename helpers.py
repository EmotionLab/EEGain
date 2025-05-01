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
from eegain.models import DeepConvNet, EEGNet, ShallowConvNet, TSception, RandomModel_class_distribution, RandomModel_most_occurring
from collections import defaultdict
from sklearn.metrics import *

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    """
    Set random seeds for reproducibility across all libraries
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    # torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    # torch.backends.cudnn.enabled = False
    
def test_one_epoch(model, loader, loss_fn, val=False):
    model.eval()
    all_preds, all_actuals = [], []
    dataset_size, running_loss, epoch_loss = 0, 0, None
    if val:
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Val ")
    else:
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
    
    return all_preds, all_actuals, epoch_loss


def run_loto(
        model,
        train_dataloader,
        val_dataloader,  # New parameter
        test_dataloader,
        test_ids,
        optimizer,
        scheduler,
        loss_fn,
        epoch,
        logger,
        random_baseline,
        subject_id,
        split_type="LOTO",
        **kwargs
):
    best_val_acc = 0.0
    best_model_state = None
    
    testID = test_ids[0]
    if isinstance(testID, str) and '<EOF>' in testID:
        testID = testID.split('<EOF>')[0]
        
    # Code to log predictions
    if kwargs.get("log_predictions", False) is True:
        prediction_dir = kwargs["log_predictions_dir"]
        # testID = test_ids[0]
        # if isinstance(testID, str) and '<EOF>' in testID:
        #     testID = testID.split('<EOF>')[0]
            
        prediction_file = os.path.join(prediction_dir, f"predictions_LOTO_VideoID{testID}.csv")
        if not os.path.exists(prediction_file):
            with open(prediction_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Data Part', 'Video id', 'Subject id', 'Actual', 'Predicted'])
    
    for i in range(epoch):
        print(f"\nEpoch {i+1}/{epoch}")
        if not random_baseline:
            # Training phase
            train_pred, train_actual, train_loss = train_one_epoch(
                model, train_dataloader, optimizer, loss_fn
            )
            
            # Validation phase
            val_pred, val_actual, val_loss = test_one_epoch(
                model, val_dataloader, loss_fn, val=True
            )
            
            # Compute validation accuracy
            val_acc = accuracy_score(val_actual, val_pred)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            # Save best model based on validation accuracy (in memory)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"New best model found with validation accuracy: {val_acc:.4f}")

            scheduler.step()
            
            #logger.log(subject_id, train_pred, train_actual, i, "train", train_loss)

        else:
            test_pred, test_actual, test_loss = test_one_epoch_random(
                model, test_dataloader, loss_fn
            )
            train_pred, train_actual, train_loss = test_pred, test_actual, test_loss
            val_pred, val_actual, val_loss = test_pred, test_actual, test_loss
            
            # Log predictions if enabled
            if kwargs.get("log_predictions", False) is True:
                with open(prediction_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for act, pred in zip(test_actual, test_pred):
                        writer.writerow([i, 'test', testID, subject_id, act, pred])
                        
            # For random model, return the regular predictions
            return train_pred, train_actual, test_pred, test_actual
        
        # Log metrics for train and validation
        logger.log(subject_id, train_pred, train_actual, i, "train", train_loss)
        
    # Load the best model for final testing (from memory)
    if not random_baseline and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Using best model for testing based on validation performance")
        
        # Final test with best model
        final_test_pred, final_test_actual, final_test_loss = test_one_epoch(
            model, test_dataloader, loss_fn
        )
        
        # Log final test results
        # testID = test_ids[0]
        # if isinstance(testID, str) and '<EOF>' in testID:
        #     testID = testID.split('<EOF>')[0]
            
        # Log predictions if enabled
        if kwargs.get("log_predictions", False) is True:
            with open(prediction_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for act, pred in zip(final_test_actual, final_test_pred):
                    writer.writerow([i, 'test', testID, subject_id, act, pred])
        
        # Return the final test results from best model
        return train_pred, train_actual, final_test_pred, final_test_actual
    
    logger.log_summary()


def run_loso(
        model,
        train_dataloader,
        val_dataloader,  # New parameter for validation
        test_dataloader,
        test_subject_ids,
        optimizer,
        scheduler,
        loss_fn,
        epoch,
        logger,
        random_baseline,
        train_videos,
        val_videos,     # New parameter for validation videos
        test_videos,
        **kwargs
):
    best_val_acc = 0.0
    best_model_state = None
        
    print(f"test subject ids {test_subject_ids}")
    
    # Code to log predictions
    if kwargs.get("log_predictions", False) is True:
        prediction_dir = kwargs["log_predictions_dir"]
        prediction_file = os.path.join(prediction_dir, f"predictions_LOSO_SubjectID{test_subject_ids[0]}.csv")
        with open(prediction_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Data Part', 'Subject id', 'Video id', 'Actual', 'Predicted'])
        
    for i in range(epoch):
        print(f"\nEpoch {i+1}/{epoch}")
        if not random_baseline:
            # Training phase
            train_pred, train_actual, train_loss = train_one_epoch(
                model, train_dataloader, optimizer, loss_fn
            )
            
            # Validation phase
            val_pred, val_actual, val_loss = test_one_epoch(
                model, val_dataloader, loss_fn, val=True
            )
            
            # Compute validation accuracy
            val_acc = accuracy_score(val_actual, val_pred)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            # Save best model based on validation accuracy (in memory)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"New best model found with validation accuracy: {val_acc:.4f}")
            
            scheduler.step()
            
            # Log metrics for train and validation
            logger.log(test_subject_ids[0], train_pred, train_actual, i, "train", train_loss)
            #logger.log(test_subject_ids[0], val_pred, val_actual, i, "val", val_loss)
        else:
            test_pred, test_actual, test_loss = test_one_epoch_random(
                model, test_dataloader, loss_fn
            )
            train_pred, train_actual, train_loss = test_pred, test_actual, test_loss
            val_pred, val_actual, val_loss = test_pred, test_actual, test_loss
            # Logging
            logger.log(test_subject_ids[0], test_pred, test_actual, i, "test", test_loss)
            # Log predictions if enabled
            if kwargs.get("log_predictions", False) is True:
                with open(prediction_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for act, pred, vid in zip(val_actual, val_pred, val_videos):
                        # for SEED dataset, the video id is in the form of 'video_id<EOF>path'
                        if isinstance(vid, str) and '<EOF>' in vid:
                            vid = vid.split('<EOF>'[0])
                        
                        writer.writerow([i, 'val', test_subject_ids[0], vid, act, pred])
              
                
        # Log metrics for train and validation
        #logger.log(test_subject_ids[0], train_pred, train_actual, i, "train", train_loss)
        #logger.log(test_subject_ids[0], val_pred, val_actual, i, "val", val_loss)
    
    # Load the best model for final testing (from memory)
    if not random_baseline and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Using best model for testing based on validation performance")
    
        # Final test phase with best model
        final_test_pred, final_test_actual, final_test_loss = test_one_epoch(
            model, test_dataloader, loss_fn
        )
    
        # Log test results
        logger.log(test_subject_ids[0], final_test_pred, final_test_actual, epoch, "test", final_test_loss)
    
        # Log test predictions if enabled
        if kwargs.get("log_predictions", False) is True:
            with open(prediction_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for act, pred, vid in zip(final_test_actual, final_test_pred, test_videos):
                    # for SEED dataset, the video id is in the form of 'video_id<EOF>path'
                    if isinstance(vid, str) and '<EOF>' in vid:
                        vid = vid.split('<EOF>'[0])
                    
                    writer.writerow([epoch, 'test', test_subject_ids[0], vid, act, pred])

    logger.log_summary(overal_log_file="overal_log", log_dir="logs/")


def main_loto(dataset, model, empty_model, classes, **kwargs):
    subject_video_mapping = dataset.mapping_list
    logger = EmotionLogger(log_dir=kwargs["log_dir"], class_names=classes)

    for subject_id, session_ids in subject_video_mapping.items():
        n_fold=len(session_ids)
        #n_fold=10 # (for 10-fold cross validation to replicate the TSception paper) 
        
        # Pass train_val_split parameter to the data loader
        train_val_split = kwargs.get("train_val_split", 0.8)
        eegloader = EEGDataloader(dataset, batch_size=kwargs["batch_size"]).loto(
            subject_id, session_ids, n_fold=n_fold, train_val_split=train_val_split)
        
        if kwargs["model_name"]=="RANDOM_most_occurring" or kwargs["model_name"]=="RANDOM_class_distribution":
            num_epoch = 1
        else:
            num_epoch = kwargs["num_epochs"]
        #num_epoch = kwargs["num_epochs"]
        
        all_train_preds_for_subject = []
        all_train_actuals_for_subject = []
        all_test_preds_for_subject = []
        all_test_actuals_for_subject = []
        
        for i, loader in enumerate(eegloader):
            if kwargs["model_name"] == "RANDOM_most_occurring":
                model = RandomModel_most_occurring(loader["train"], loader["val"])
                is_random = True
                optimizer = None
                scheduler = None
            elif kwargs["model_name"] == "RANDOM_class_distribution":
                model = RandomModel_class_distribution(loader["train"], loader["val"])
                is_random = True
                optimizer = None
                scheduler = None
            else:
                model = copy.deepcopy(empty_model)
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
                is_random = False
                
            loss_fn = nn.CrossEntropyLoss(label_smoothing=kwargs["label_smoothing"])
            
            # Call the modified run_loto with validation data
            train_pred, train_actual, test_pred, test_actual = run_loto(
                model=model,
                train_dataloader=loader["train"],
                val_dataloader=loader["val"],  # Pass validation dataloader
                test_dataloader=loader["test"],
                test_ids=loader["test_session_indexes"],
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                epoch=num_epoch,
                logger=logger,
                random_baseline=is_random,
                subject_id=loader["subject_id"],
                **kwargs,
            )
            
            # all_train_preds_for_subject.append(train_pred)
            # all_train_actuals_for_subject.append(train_actual)
            all_test_preds_for_subject.append(test_pred)
            all_test_actuals_for_subject.append(test_actual)

        # Flatten lists of predictions and actuals
        # all_train_preds_for_subject = [item for sublist in all_train_preds_for_subject for item in sublist]
        # all_train_actuals_for_subject = [item for sublist in all_train_actuals_for_subject for item in sublist]
        all_test_preds_for_subject = [item for sublist in all_test_preds_for_subject for item in sublist]
        all_test_actuals_for_subject = [item for sublist in all_test_actuals_for_subject for item in sublist]

        logger.log(subject_id, all_test_preds_for_subject, all_test_actuals_for_subject, num_epoch, "test")
        
    logger.log_summary(overal_log_file=kwargs["overal_log_file"], log_dir=kwargs["log_dir"])


def loso_loop(model, loader, logger, **kwargs):
    if kwargs["model_name"]=="RANDOM_most_occurring":
        model = RandomModel_most_occurring(loader["train"], loader["val"])
        optimizer = None
        is_random = True
        scheduler = None
    elif kwargs["model_name"]=="RANDOM_class_distribution":
        model = RandomModel_class_distribution(loader["train"], loader["val"])
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
    
    if kwargs["model_name"]=="RANDOM_most_occurring" or kwargs["model_name"]=="RANDOM_class_distribution":
        num_epoch = 1
    else:
        num_epoch = kwargs["num_epochs"]
        
    run_loso(
        model=model,
        train_dataloader=loader["train"],
        val_dataloader=loader["val"],  # Add validation dataloader
        test_dataloader=loader["test"],
        test_subject_ids=loader["test_subject_indexes"],
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epoch=num_epoch,
        logger=logger,
        random_baseline=is_random,
        train_videos = loader["train_videos"],  # new params
        val_videos = loader["val_videos"],  # Add validation videos
        test_videos = loader["test_videos"],    # new params
        **kwargs
    )



def main_loso(dataset, model, empty_model, classes, **kwargs):
    #eegloader = EEGDataloader(dataset, batch_size=kwargs["batch_size"]).loso()
    train_val_split = kwargs.get("train_val_split", 0.8)
    eegloader = EEGDataloader(dataset, batch_size=kwargs["batch_size"]).loso(train_val_split=train_val_split)

    logger = EmotionLogger(log_dir=kwargs["log_dir"], class_names=classes)
    for loader in eegloader:
        if kwargs["model_name"]=="RANDOM_most_occurring":
            model = RandomModel_most_occurring(loader["train"], loader["val"])
        elif kwargs["model_name"]=="RANDOM_class_distribution":
            model = RandomModel_class_distribution(loader["train"], loader["val"])
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

    # Pass train_val_split parameter to the data loader
    train_val_split = kwargs.get("train_val_split", 0.8)
    eegloader = EEGDataloader(dataset, batch_size=kwargs["batch_size"]).loso_fixed(
        train_set, test_set, train_val_split=train_val_split)

    logger = EmotionLogger(log_dir=kwargs["log_dir"], class_names=classes)
    if kwargs["model_name"]=="RANDOM_most_occurring":
        model = RandomModel_most_occurring(eegloader["train"], eegloader["val"])
    elif kwargs["model_name"]=="RANDOM_class_distribution":
        model = RandomModel_class_distribution(eegloader["train"], eegloader["val"])
    
    loso_loop(model, eegloader, logger, **kwargs)
    logger.log_summary(overal_log_file=kwargs["overal_log_file"], log_dir=kwargs["log_dir"])