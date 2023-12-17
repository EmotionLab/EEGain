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
    logger
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

    logger.log_summary()


@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


def generate_options():
    def decorator(func):
        config_instances = [TransformConfig, DataConfig, TrainingConfig, EEGNetConfig, TScepctionConfig]
        for config_instance in config_instances:
            for field, value in asdict(config_instance()).items():
                option = click.option(f"--{field}", default=value, required=False, type=type(value))
                func = option(func)
        return func
    return decorator



@click.command()
@click.option("--model_name", required=True, type=str, help="name of the config")
@generate_options()
def main(**kwargs):
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
            eegain.transforms.Resample(s_rate=kwargs["s_rate"]),
            eegain.transforms.Segment(duration=kwargs["window"], overlap=0),
        ]
    )

    # -------------- Dataset --------------
    mahnob_dataset = MAHNOB(
        kwargs["data_path"],
        label_type=kwargs["label_type"],
        transform=transform,
    )

    # -------------- Dataloader --------------
    eegloader = EEGDataloader(mahnob_dataset, batch_size=kwargs["batch_size"]).loso()  # .loto()

    # -------------- Training --------------
    logger = EmotionLogger(log_dir=kwargs["log_dir"], class_names=["low", "high"])
    for loader in eegloader:
        # -------------- Model --------------
        model = globals()[kwargs['model_name']](input_size=[1, 32, kwargs["window"]*kwargs["s_rate"]], **kwargs)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
        loss_fn = nn.CrossEntropyLoss()
        run(
            model=model,
            train_dataloader=loader["train"],
            test_dataloader=loader["test"],
            test_subject_ids=loader["test_subject_indexes"],  # loader["test_video_indexes"] - for loto
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=kwargs["num_epochs"],
            logger=logger,
        )


if __name__ == "__main__":
    main()

