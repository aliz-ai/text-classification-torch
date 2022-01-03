import os
import pickle

from google.cloud import storage
from typing import Any
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import numpy as np
import pickle
from torch import nn
from datasets import load_metric
import os
from google.cloud import storage
import torch


def create_training_arguments(args):
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


# Create loss function (CrossEntropyLoss) for multiclass classification
class MulticlassTrainer(Trainer):
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
                    if (
                        best_val_loss is None
                        or log["eval_loss"] < best_val_loss
                    ):
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
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(bucket_uri)
    blob.upload_from_filename(temp_file)
