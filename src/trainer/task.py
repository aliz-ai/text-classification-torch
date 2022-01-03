import argparse

import pandas as pd
import google.cloud.logging
import logging
from sklearn.preprocessing import LabelEncoder
from transformers import AutoConfig, AutoModelForSequenceClassification, EarlyStoppingCallback
from trainer.utils import (
    create_training_arguments,
    SaveModelCallback,
    MulticlassTrainer,
    compute_metrics,
    get_label_ids,
    pickle_to_bucket
)
from trainer.data import TextDataset

if __name__ == "__main__":
    try:
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--project",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--bucket",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--data",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--output-path",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--epochs",
            type=int,
            required=True,
        )
        parser.add_argument(
            "--fp16",
            type=bool,
            default=False
        )
        parser.add_argument(
            "--early-stopping-patience",
            type=int,
            default=1
        )
        parser.add_argument(
            "--eval-steps",
            type=int,
            default=3000
        )
        parser.add_argument("--batch-size", default=16, type=int)
        parser.add_argument("--bert-version", default="bert-base-cased", type=str)
        args = parser.parse_args()

        # Setup logging
        loggin_client = google.cloud.logging.Client(project=args.project)
        loggin_client.get_default_handler()
        loggin_client.setup_logging()

        logging.info(args.__dict__)

        # Load training data
        logging.info("Load data")
        data = pd.read_csv(args.data, header=None, names=["ml_use", "text", "label"])

        # Encoder labels
        logging.info("Encoder labels")
        label_encoder = LabelEncoder()
        label_encoder.fit(data[data.ml_use == "training"].label)
        data["y"] = label_encoder.transform(data.label)

        # Create Torch datasets
        logging.info("Create datasets")
        train_dataset = TextDataset(
            texts=data[data.ml_use == "training"].text.values,
            labels=data[data.ml_use == "training"].y.values,
            bert_version=args.bert_version,
        )
        valid_dataset = TextDataset(
            texts=data[data.ml_use == "validation"].text.values,
            labels=data[data.ml_use == "validation"].y.values,
            bert_version=args.bert_version,
        )

        # Label-to-id conversion dictionaries
        logging.info("Save label-id conversion files")
        label2id, id2label = get_label_ids(label_encoder)
        pickle_to_bucket(
            project_name=args.project,
            bucket_name=args.bucket,
            bucket_uri=f"{args.output_path}/label_id.pkl",
            object=(label2id, id2label),
        )

        # Create model
        logging.info("Create model")
        config = AutoConfig.from_pretrained(
            args.bert_version, label2id=label2id, id2label=id2label
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.bert_version, config=config
        )

        # Train model
        logging.info("Set up trainer")
        training_args = create_training_arguments(args)
        trainer = MulticlassTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.add_callback(SaveModelCallback(args, model))
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        
        logging.info("Start training")
        trainer.train()
    except Exception as e:
        logging.error(e)
        assert 1 == 2

