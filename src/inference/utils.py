import pickle
from typing import Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from trainer.utils import download_file_from_bucket


def create_model(bert_version: str, label2id: Dict, id2label: Dict):
    config = AutoConfig.from_pretrained(
        bert_version, label2id=label2id, id2label=id2label
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_version, config=config
    )
    return model


def load_model(
    model,
    bucket: str,
    model_path: str,
    device,
    project: str = None,
):
    local_path = "temp/model.pt"
    download_file_from_bucket(project, bucket, model_path, local_path)

    state = torch.load(local_path, map_location=device)

    model.load_state_dict(state)
    model.eval()
    model = model.to(device)
    return model


def load_label_id(project: str, bucket: str, label_id_path: str) -> Tuple[Dict, Dict]:
    local_path = "temp/label_id.pkl"
    download_file_from_bucket(project, bucket, label_id_path, local_path)

    with open(local_path, "rb") as f:
        label2id, id2label = pickle.load(f)

    return label2id, id2label
