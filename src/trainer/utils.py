import os
import pickle
from typing import Any

import google.cloud.logging
import numpy as np
import torch
from datasets import load_metric
from google.cloud import storage
from torch import nn
from transformers import Trainer, TrainerCallback, TrainingArguments


def create_training_arguments(args):
    """Sets up required training arguments."""

    training_args = TrainingArguments(
        output_dir=f"{args.output_path}/model",
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=args.fp16,
        warmup_steps=1000,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        num_train_epochs=args.epochs,
        dataloader_num_workers=0,
        load_best_model_at_end=True,
        logging_dir=f"gs://{args.bucket}/{args.output_path}/logs",
        logging_steps=50,
        report_to="tensorboard",
    )
    return training_args


class MulticlassTrainer(Trainer):
    """Create loss function (CrossEntropyLoss) for multiclass classification"""

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1).long(),
        )
        return (loss, outputs) if return_outputs else loss


metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class SaveModelCallback(TrainerCallback):
    """Saves model checkpoint to GCS bucket"""

    def __init__(self, training_args, model, *args, **kwargs) -> None:
        super().__init__(*args, *kwargs)
        self.args = training_args
        self.model = model

    def on_evaluate(self, _, state, control, logs=None, **kwargs):
        def get_best_step(state):
            best_step = None
            best_val_loss = None
            for log in state.log_history:
                if "eval_loss" in log:
                    if best_val_loss is None or log["eval_loss"] < best_val_loss:
                        best_val_loss = log["eval_loss"]
                        best_step = log["step"]
            return best_step

        os.makedirs("temp", exist_ok=True)
        storage_client = storage.Client()
        with open("temp/state.pkl", "wb") as f:
            pickle.dump(state, f)

        bucket = storage_client.bucket(self.args.bucket)
        blob = bucket.blob(f"{self.args.output_path}/state.pkl")
        blob.upload_from_filename("temp/state.pkl")

        if state.global_step == get_best_step(state):
            torch.save(self.model.state_dict(), "temp/model.pt")
            blob = bucket.blob(f"{self.args.output_path}/model.pt")
            blob.upload_from_filename("temp/model.pt")


def get_label_ids(encoder):
    """Creates label-id transformation dictionaries from class names."""
    label2id = {}
    id2label = {}
    for id, label in enumerate(encoder.classes_):
        label2id[label] = id
        id2label[id] = label

    return label2id, id2label


def pickle_to_bucket(
    project_name: str, bucket_name: str, bucket_uri: str, object: Any
) -> None:
    """
    Uploads any object to a bucket as a pickle dump.
    """
    os.makedirs("temp", exist_ok=True)
    temp_file = "temp/temp.pkl"
    with open(temp_file, "wb") as f:
        pickle.dump(object, f)

    upload_file_to_bucket(project_name, bucket_name, bucket_uri, temp_file)


def download_file_from_bucket(
    project: str, bucket: str, file_path: str, local_path: str
):
    storage_client = storage.Client(project)
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(file_path)
    os.makedirs("temp", exist_ok=True)
    blob.download_to_filename(local_path)


def upload_file_to_bucket(project: str, bucket: str, file_path: str, local_path: str):
    storage_client = storage.Client(project)
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(file_path)
    blob.upload_from_filename(local_path)


def setup_cloud_logging(project):
    """
    Sets up cloud logging.
    You can use the regular logging library to log everything as you did before locally.
    Check your logs with Logs Explorer on Console.
    """

    loggin_client = google.cloud.logging.Client(project=project)
    loggin_client.get_default_handler()
    loggin_client.setup_logging()
