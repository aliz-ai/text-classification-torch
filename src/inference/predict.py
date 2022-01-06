import argparse
import logging
import os
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoTokenizer, TextClassificationPipeline

from inference.utils import create_model, load_label_id, load_model
from trainer.utils import setup_cloud_logging, upload_file_to_bucket


def predict(texts: List[str], args: Dict):
    # Load label encoders
    logging.info("Loading label2id converters...")
    label2id, id2label = load_label_id(args.project, args.bucket, args.label_id_path)

    # Load trained model
    logging.info("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.bert_version, label2id, id2label)
    model = load_model(
        bucket=args.bucket,
        project=args.project,
        model_path=args.model_path,
        model=model,
        device=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.bert_version)

    # Create prediction pipeline
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        truncation=True,
        device=0 if torch.cuda.is_available() else -1,
    )

    # Predict
    preds = [out["label"] for out in pipe(texts, batch_size=args.batch_size)]
    return preds


def save_preds(preds: List[str], args: Dict):
    os.makedirs("temp", exist_ok=True)
    local_path = "temp/preds.csv"
    pd.DataFrame(preds).to_csv(local_path, header=None, index=False)

    upload_file_to_bucket(
        project=args.project,
        bucket=args.bucket,
        file_path=f"{args.output_path}/preds.csv",
        local_path=local_path,
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

    # Setup logging
    setup_cloud_logging(args.project)

    # Load input
    # Use a CSV for now
    texts = pd.read_csv(args.data, header=None)[0].values.tolist()

    # Predict
    preds = predict(texts, args)

    # Save predictions
    save_preds(preds, args)
