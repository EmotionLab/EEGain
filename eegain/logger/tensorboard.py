import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import *
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("Tensorboard")


# Logger class for logging metrics for each subject or video
class Logger:
    def __init__(
        self, id: int, writer_train: SummaryWriter, writer_test: SummaryWriter, class_names: list[str]
    ):
        self.id = id
        self.writer_train = writer_train
        self.writer_test = writer_test
        self.metrics = {}
        self.class_names = class_names

    def _log_confusion_matrix(self, value: float | list, step: int, data_part: str):
        if value.shape == (1, 1):
            df_cm = pd.DataFrame(
                value, index=[0], columns=[0]
            )
        else:
            df_cm = pd.DataFrame(
                value, index=[i for i in self.class_names], columns=[i for i in self.class_names]
            )
        plt.figure(figsize=(12, 7))
        image = sn.heatmap(df_cm, annot=True).get_figure()
        plt.title("Valence")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        if data_part == "train":
            self.writer_train.add_figure(
                f"id {self.id}/confusion_matrix", image, step
            )
        else:
            self.writer_test.add_figure(
                f"id {self.id}/confusion_matrix", image, step
            )

    def log_metric(
        self, step: int, metric_name: str, value: float | list, data_part: str
    ):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        if metric_name == "confusion_matrix":
            self._log_confusion_matrix(value, step, data_part)
        else:
            if data_part == "train":
                self.writer_train.add_scalar(
                    f"id {self.id}/{metric_name}", value, step
                )
            else:
                self.writer_test.add_scalar(
                    f"id {self.id}/{metric_name}", value, step
                )

    def get_subject_metrics(self):
        return self.metrics


# EmotionLogger class for handling the overall logging functionality
class EmotionLogger:
    def __init__(self, log_dir: str, class_names: int):
        self.loggers = {}
        self.loto_logger = {}
        self.loto_logger_overall = {}
        self.writer = SummaryWriter(log_dir)
        self.writer_train = SummaryWriter(f"{log_dir}/train")
        self.writer_test = SummaryWriter(f"{log_dir}/test")
        self.class_names = class_names
        self.num_class = len(class_names)

        logger.info(f"Using Tensorboard logger")
        logger.info(
            "Logging: Accuracy, F1, Precision, Recall, Kappa, matthews corr coef, confusion matrix"
        )
        logger.info(f"Saving logs in {log_dir}")

    def add_logger(self, id: int):
        if id not in self.loggers:
            self.loggers[id] = Logger(
                id, self.writer_train, self.writer_test, self.class_names
            )

    def add_loto_logger(self, id: int):
        if id not in self.loto_logger:
            self.loto_logger[id] = Logger(
                id, self.writer_train, self.writer_test, self.class_names
            )

    def log_metric(
        self,
        id: int,
        metric_name: str,
        value: float | list,
        step: int,
        data_part: str,
        split_type: str,
    ):
        id_type = ""
        if split_type.upper() == "LOSO":
            id_type = "subject"
            if id not in self.loggers:
                self.add_logger(id)
            self.loggers[id].log_metric(step, metric_name, value, data_part)
        elif split_type.upper() == "LOTO":
            id_type = "video"
            if id not in self.loto_logger:
                self.add_loto_logger(id)
            self.loto_logger[id].log_metric(step, metric_name, value, data_part)

        if metric_name != "confusion_matrix":
            logger.info(
                f"({data_part}) {id_type}_id:{id} {metric_name}={value:.4f} step={step}"
            )
        else:
            logger.info(
                f"({data_part}) {id_type}_id:{id} {metric_name}={value} step={step}"
            )

    def log(
        self,
        id: int,
        test_pred: torch.Tensor,
        test_actual: torch.Tensor,
        i: int,
        data_part: str,
        split_type: str,
        loss=None,
    ):

        self.log_metric(
            id, "accuracy",
            accuracy_score(test_actual, test_pred), i, data_part, split_type
        )
        self.log_metric(
            id, "f1",
            f1_score(test_actual, test_pred, average='binary' if self.num_class <= 2 else 'weighted'), i, data_part, split_type
        )
        self.log_metric(
            id, "recall",
            recall_score(test_actual, test_pred, average='binary' if self.num_class <= 2 else 'weighted'), i, data_part, split_type
        )
        self.log_metric(
            id,
            "precision",
            precision_score(test_actual, test_pred, average='binary' if self.num_class <= 2 else 'weighted'), i, data_part, split_type
        )
        self.log_metric(
            id, "kappa",
            cohen_kappa_score(test_actual, test_pred), i, data_part, split_type
        )
        # self.log_metric(
        #     subject_id, "roc_auc",
        #     roc_auc_score(test_actual, test_pred, average='binary' if self.num_class <= 2 else 'weighted', multi_class='ovr'), i, data_part
        # )
        self.log_metric(
            id,
            "matthews_corrcoef",
            matthews_corrcoef(test_actual, test_pred), i, data_part, split_type
        )
        self.log_metric(
            id,
            "confusion_matrix",
            confusion_matrix(test_actual, test_pred), i, data_part, split_type
        )
        if loss:
            self.log_metric(
                id,
                "loss",
                loss, i, data_part, split_type
            )

    def log_each_user_metrics(self, metric_names: list[str]):
        metrics_for_each_subject = {}

        for id, subject_logger in self.loggers.items():
            metrics_for_each_subject[id] = subject_logger.metrics

        for metric_name in metric_names:
            x = metrics_for_each_subject.keys()
            y = [metrics_for_each_subject[i][metric_name][-1] for i in x]

            plt.bar(x, y)
            plt.xlabel("Metric")
            plt.ylabel(metric_name)

            self.writer.add_figure(f"For each subject/{metric_name}", plt.gcf())
            plt.close()

    def log_overall_metrics(self, metric_names: list[str]):
        metrics_for_each_subject = {}
        for id, subject_logger in self.loggers.items():
            metrics_for_each_subject[id] = subject_logger.metrics

        for metric_name in metric_names:
            metric_average = []
            for _, metrics in metrics_for_each_subject.items():
                metric_average.append(metrics[metric_name][-1])

            metric_average = sum(metric_average) / len(metric_average)

            self.writer.add_scalar(f"Overall/{metric_name}", metric_average)

    def log_each_user_metrics_loto(self, metric_names: list[str], subject_id: int = None):
        metrics_for_each_video = {}

        for id, subject_logger in self.loto_logger.items():
            metrics_for_each_video[id] = subject_logger.metrics

        for metric_name in metric_names:
            x = metrics_for_each_video.keys()
            y = [metrics_for_each_video[i][metric_name][-1] for i in x]

            plt.bar(x, y)
            plt.xlabel("Metric")
            plt.ylabel(metric_name)

            self.writer.add_figure(f"For subject {subject_id} all video/{metric_name}", plt.gcf())
            plt.close()

    def log_overall_loto_for_each_subject(self, metric_names: list[str], subject_id: int):
        metrics_for_each_subject = {}
        for id, subject_logger in self.loto_logger.items():
            metrics_for_each_subject[id] = subject_logger.metrics

        for metric_name in metric_names:
            metric_average = []
            for _, metrics in metrics_for_each_subject.items():
                metric_average.append(metrics[metric_name][-1])

            metric_average = sum(metric_average) / len(metric_average)

            if metric_name in self.loto_logger_overall.keys():
                self.loto_logger_overall[metric_name].append(metric_average)
            else:
                self.loto_logger_overall[metric_name] = [metric_average]

            self.writer.add_scalar(f"Overall for subject {subject_id}/{metric_name}", metric_average)

    def log_overall_loto(self, metric_names: list[str], subject_id: int):
        metrics_for_each_subject = {}
        for id, subject_logger in self.loto_logger_overall.items():
            metrics_for_each_subject[id] = subject_logger.metrics

        for metric_name in metric_names:
            metric_average = []
            for _, metrics in metrics_for_each_subject.items():
                metric_average.append(metrics[metric_name][-1])

            metric_average = sum(metric_average) / len(metric_average)

            self.writer.add_scalar(f"Overall/{metric_name}", metric_average)

    def log_summary(self, split_type=None, subject_id=None):
        if split_type == "LOTO":
            self.log_each_user_metrics_loto(
                [
                    "accuracy",
                    "f1",
                    "recall",
                    "precision",
                    "kappa",
                    # "roc_auc",
                    "matthews_corrcoef",
                ],
                subject_id
            )
            self.log_overall_loto_for_each_subject(
                [
                    "accuracy",
                    "f1",
                    "recall",
                    "precision",
                    "kappa",
                    # "roc_auc",
                    "matthews_corrcoef",
                ],
                subject_id
            )
        else:
            self.log_each_user_metrics(
                [
                    "accuracy",
                    "f1",
                    "recall",
                    "precision",
                    "kappa",
                    # "roc_auc",
                    "matthews_corrcoef",
                ]
            )
            self.log_overall_metrics(
                [
                    "accuracy",
                    "f1",
                    "recall",
                    "precision",
                    "kappa",
                    # "roc_auc",
                    "matthews_corrcoef",
                ]
            )
            self.writer.close()
