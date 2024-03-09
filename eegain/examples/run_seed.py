import os

os.environ["LOG_LEVEL"] = "DEBUG"

import torch
from torch import nn
from tqdm import tqdm

import eegain
from eegain.data import EEGDataloader
from eegain.data.datasets import DEAP, MAHNOB, Seed
from eegain.logger import EmotionLogger
from eegain.models import DeepConvNet, EEGNet, ShallowConvNet, TSception


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
    test_subject_ids,
    optimizer,
    loss_fn,
    epoch,
):
    for i in range(epoch):
        print(f"\nEpoch {i}/{epoch}")

        train_pred, train_actual, train_loss = train_one_epoch(
            model, train_dataloader, optimizer, loss_fn
        )
        test_pred, test_actual, test_loss = test_one_epoch(
            model, test_dataloader, loss_fn
        )

        logger.log(test_subject_ids[0], train_pred, train_actual, i, "train", train_loss)
        logger.log(test_subject_ids[0], test_pred, test_actual, i, "val", test_loss)

    logger.log_summary(overal_log_file="overal_log", log_dir="logs/")


# -------------- Preprocessing --------------
channels_to_drop_seed_iv = [
    "FPZ",
    "F5",
    "F1",
    "F2",
    "F6",
    "FT7",
    "FC3",
    "FCZ",
    "FC4",
    "FT8",
    "C5",
    "C1",
    "C2",
    "C6",
    "TP7",
    "CP3",
    "CPZ",
    "CP4",
    "TP8",
    "P5",
    "P1",
    "PZ",
    "P2",
    "P6",
    "PO7",
    "PO5",
    "PO6",
    "PO8",
    "CB1",
    "CB2",
]

transform = eegain.transforms.Construct(
    [
        eegain.transforms.DropChannels(channels_to_drop_seed_iv),
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(s_rate=128),
        eegain.transforms.Segment(duration=4, overlap=0),
    ]
)


# -------------- Dataset --------------
seed_dataset = Seed(
    "/Users/rango/Desktop/GAIN/SEED",
    label_type="V",
    transform=transform,
)


# -------------- Dataloader --------------
eegloader = EEGDataloader(seed_dataset, batch_size=32).loso()


# -------------- Training --------------
logger = EmotionLogger(log_dir="logs/", class_names=[0, 1, 2, 3])
for loader in eegloader:
    # -------------- Model --------------
    model = TSception(
        num_classes=3,
        input_size=(1, 32, 512),
        sampling_r=128,
        num_t=15,
        num_s=15,
        hidden=32,
        dropout_rate=0.5,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    run(
        model=model,
        train_dataloader=loader["train"],
        test_dataloader=loader["test"],
        test_subject_ids=loader["test_subject_indexes"],
        optimizer=optimizer,
        loss_fn=loss_fn,
        epoch=3,
    )
