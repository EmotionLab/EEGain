import os

os.environ["LOG_LEVEL"] = "DEBUG"

import torch
from torch import nn
from tqdm import tqdm

import eegain
from eegain.data import EEGDataloader
from eegain.data.datasets import DEAP, MAHNOB, SeedIV
from eegain.logger import EmotionLogger
from eegain.models import DeepConvNet, EEGNet, ShallowConvNet, TSception
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


def run(
    model,
    train_dataloader,
    test_dataloader,
    test_ids,
    optimizer,
    loss_fn,
    epoch,
    split_type="LOTO",
):

    for i in range(epoch):
        print(f"\nEpoch {i}/{epoch}")

        train_pred, train_actual, train_loss = train_one_epoch(
            model, train_dataloader, optimizer, loss_fn
        )
        test_pred, test_actual, test_loss = test_one_epoch(
            model, test_dataloader, loss_fn
        )
        if split_type != "LOTO":
            # in the end of epoch it logs metrics for this specific epoch. test_ids is test_session_ids
            logger.log(test_ids[0], train_pred, train_actual, i, "train", train_loss)
            logger.log(test_ids[0], test_pred, test_actual, i, "val", test_loss)
    if split_type == "LOTO": 
        return train_pred, train_actual, test_pred, test_actual
    # for loso
    logger.log_summary()


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
    "path_to_mahnob",
    label_type="V",
    transform=transform,
)

subject_video_mapping = mahnob_dataset.mapping_list
logger = EmotionLogger(log_dir="logs/", class_names=["low", "high"])

all_model_state_dicts = []
all_train_preds, all_test_preds, all_train_actuals, all_test_actuals = [], [], [], []
f1_tests, f1_weighted_tests, accuracy_tests = [], [], []

for subject_id, session_ids in subject_video_mapping.items():
    eegloader = EEGDataloader(mahnob_dataset, batch_size=32).loto(subject_id, session_ids, n_fold=10) # pass n_fold=len(session_ids) for LOTO
    num_epoch = 5
    all_train_preds_for_subject, all_train_actuals_for_subject, all_test_preds_for_subject, all_test_actuals_for_subject = [], [], [], []
    for i, loader in enumerate(eegloader):
        model = TSception(
                        num_classes=2,
                        input_size=(1, 32, 512),
                        sampling_r=128,
                        num_t=15,
                        num_s=15,
                        hidden=32,
                        dropout_rate=0.5,
                        )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
        loss_fn = nn.CrossEntropyLoss()
        _, _, test_pred, test_actual = run(
            model=model,
            train_dataloader=loader["train"],
            test_dataloader=loader["test"],
            test_ids=loader["test_session_indexes"],
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=num_epoch,
            split_type="LOTO"  # for loto
        )
        all_test_preds_for_subject.append(test_pred)
        all_test_actuals_for_subject.append(test_actual)

    # all_model_state_dicts.append(model.state_dict())
    all_test_preds_for_subject = [item for sublist in all_test_preds_for_subject for item in sublist]
    all_test_actuals_for_subject = [item for sublist in all_test_actuals_for_subject for item in sublist]
        
    logger.log(subject_id, all_test_preds_for_subject, all_test_actuals_for_subject, num_epoch, "val")

logger.log_summary(overal_log_file="overal_log", log_dir="logs/")

# average_state_dict = {}
# for key in all_model_state_dicts[0]:
#     element_sum = sum(model_state_dict[key] for model_state_dict in all_model_state_dicts)
#     average_state_dict[key] = element_sum / len(all_model_state_dicts)

# average_model = TSception(
#         num_classes=2,
#         input_size=(1, 32, 512),
#         sampling_r=128,
#         num_t=15,
#         num_s=15,
#         hidden=32,
#         dropout_rate=0.5,
#     )
# average_model.load_state_dict(average_state_dict)
# average_model = average_model.to(device)

