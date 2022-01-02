import os
import pickle

from google.cloud import storage
from typing import Any


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
