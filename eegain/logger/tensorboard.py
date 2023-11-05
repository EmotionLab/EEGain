import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import *
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("Tensorboard")


# SubjectLogger class for logging metrics for each subject
class SubjectLogger:
    def __init__(
        self, subject_id: int, writer_train: SummaryWriter, writer_test: SummaryWriter, class_names: list[str]
    ):
        self.subject_id = subject_id
        self.writer_train = writer_train
        self.writer_test = writer_test
        self.metrics = {}
        self.class_names = class_names

    def _log_confusion_matrix(self, value: float | list, step: int, data_part: str):
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
                f"Subject{self.subject_id}/confusion_matrix", image, step
            )
        else:
            self.writer_test.add_figure(
                f"Subject{self.subject_id}/confusion_matrix", image, step
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
                    f"Subject{self.subject_id}/{metric_name}", value, step
                )
            else:
                self.writer_test.add_scalar(
                    f"Subject{self.subject_id}/{metric_name}", value, step
                )

    def get_subject_metrics(self):
        return self.metrics


# EmotionLogger class for handling the overall logging functionality
class EmotionLogger:
    def __init__(self, log_dir: str, class_names: int):
        self.subject_loggers = {}
        self.writer = SummaryWriter(log_dir)
        self.writer_train = SummaryWriter(f"{log_dir}/train")
        self.writer_test = SummaryWriter(f"{log_dir}/test")
        self.class_names = class_names
        self.num_class = len(class_names)

        logger.info(f"Using Tensorboard logger")
        logger.info(
            "Logging: Accuracy, F1, Precision, Recall, Kappa, Roc_auc, matthews corr coef, confusion matrix"
        )
        logger.info(f"Saving logs in {log_dir}")

    def add_subject_logger(self, subject_id: int):
        if subject_id not in self.subject_loggers:
            self.subject_loggers[subject_id] = SubjectLogger(
                subject_id, self.writer_train, self.writer_test, self.class_names
            )

    def log_metric(
        self,
        subject_id: int,
        metric_name: str,
        value: float | list,
        step: int,
        data_part: str,
    ):
        if subject_id not in self.subject_loggers:
            self.add_subject_logger(subject_id)

        self.subject_loggers[subject_id].log_metric(step, metric_name, value, data_part)

        if metric_name != "confusion_matrix":
            logger.info(
                f"({data_part}) subject_id:{subject_id} {metric_name}={value:.4f} step={step}"
            )
        else:
            logger.info(
                f"({data_part}) subject_id:{subject_id} {metric_name}={value} step={step}"
            )

    def log(
        self,
        subject_id: int,
        test_pred: torch.Tensor,
        test_actual: torch.Tensor,
        i: int,
        data_part: str,
        loss=None,
    ):

        self.log_metric(
            subject_id, "accuracy",
            accuracy_score(test_actual, test_pred), i, data_part
        )
        self.log_metric(
            subject_id, "f1",
            f1_score(test_actual, test_pred, average='binary' if self.num_class <= 2 else 'weighted'), i, data_part
        )
        self.log_metric(
            subject_id, "recall",
            recall_score(test_actual, test_pred, average='binary' if self.num_class <= 2 else 'weighted'), i, data_part
        )
        self.log_metric(
            subject_id,
            "precision",
            precision_score(test_actual, test_pred, average='binary' if self.num_class <= 2 else 'weighted'), i, data_part
        )
        self.log_metric(
            subject_id, "kappa",
            cohen_kappa_score(test_actual, test_pred), i, data_part
        )
        # self.log_metric(
        #     subject_id, "roc_auc",
        #     roc_auc_score(test_actual, test_pred, average='binary' if self.num_class <= 2 else 'weighted', multi_class='ovr'), i, data_part
        # )
        self.log_metric(
            subject_id,
            "matthews_corrcoef",
            matthews_corrcoef(test_actual, test_pred), i, data_part
        )
        self.log_metric(
            subject_id,
            "confusion_matrix",
            confusion_matrix(test_actual, test_pred), i, data_part
        )
        if loss:
            self.log_metric(
                subject_id,
                "loss",
                loss, i, data_part
            )

    def log_each_user_metrics(self, metric_names: list[str]):
        metrics_for_each_subject = {}
        for subject_id, subject_logger in self.subject_loggers.items():
            metrics_for_each_subject[subject_id] = subject_logger.metrics

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
        for subject_id, subject_logger in self.subject_loggers.items():
            metrics_for_each_subject[subject_id] = subject_logger.metrics

        for metric_name in metric_names:
            metric_average = []
            for _, metrics in metrics_for_each_subject.items():
                metric_average.append(metrics[metric_name][-1])

            metric_average = sum(metric_average) / len(metric_average)

            self.writer.add_scalar(f"Overall/{metric_name}", metric_average)

    def log_summary(self):
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
