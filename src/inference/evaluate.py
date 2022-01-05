from transformers.pipelines import pipeline
import logging
import argparse
from typing import List
from inference.predict import predict
from trainer.utils import upload_file_to_bucket
from sklearn.metrics import classification_report
import os
from trainer.utils import setup_cloud_logging
import pandas as pd


def evaluate(y_true, y_pred, args):
    # Evaluate
    report = classification_report(y_true, y_pred)

    # Save results
    os.makedirs("temp", exist_ok=True)
    local_path = "temp/report.txt"
    with open(local_path, "w") as f:
        f.write(report)
    
    upload_file_to_bucket(
        project=args.project,
        bucket=args.bucket,
        file_path=f"{args.output_path}/report.txt",
        local_path=local_path
    )


if __name__ == "__main__":
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
        "--model-path",
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
        "--label-id-path",
        type=str,
        required=True,
    )
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--bert-version", default="bert-base-cased", type=str)
    args = parser.parse_args()


    # Setup cloud logging
    setup_cloud_logging(args.project)
    

    # Load test set
    data = pd.read_csv(args.data)
    labels = data[data.iloc[:, 0] == "test"].iloc[:, 2].tolist()
    texts = data[data.iloc[:, 0] == "test"].iloc[:, 1].tolist()

    # Predict
    preds = predict(texts, args)

    # Evaluation
    evaluate(labels, preds, args)