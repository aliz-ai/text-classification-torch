from datetime import datetime
from typing import Optional, Union

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip


def run_training_job(
    data: str,
    epochs: Union[float, int],
    project: str,
    bucket: str,
    output_path: str,
    batch_size: int,
    eval_steps: int,
    region: str,
    api_endpoint: str,
    package_uri: str,
    # service_account: str,
    n_gpu: int = 4
):
    """
    :param model_datetime: Model ID of previously trained model.
    If None a general multilingual pretrained BERT will be loaded.
    :param model_step: Checkpoint of the model to be loaded.
    """
    TRAIN_GPU, TRAIN_NGPU = (aip.AcceleratorType.NVIDIA_TESLA_V100, n_gpu)
    TRAIN_IMAGE = "europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:m82"
    TRAIN_COMPUTE = "n1-standard-4"
    TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    JOB_NAME = "train_text_classifier_" + TIMESTAMP
    MODEL_DISPLAY_NAME = JOB_NAME

    CMDARGS = [
        "--epochs=" + str(epochs),
        "--bucket=" + bucket,
        "--project=" + project,
        "--data=" + data,
        "--output-path=" + output_path,
        "--batch-size=" + str(batch_size),
        "--epochs=" + str(epochs),
        "--eval-steps=" + str(eval_steps)
    ]

    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    custom_job = {
        "display_name": MODEL_DISPLAY_NAME,
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": TRAIN_COMPUTE,
                        "accelerator_type": TRAIN_GPU,
                        "accelerator_count": TRAIN_NGPU,
                    },
                    "replica_count": 1,
                    "python_package_spec": {
                        "executor_image_uri": TRAIN_IMAGE,
                        "package_uris": [package_uri],
                        "python_module": "trainer.task",
                        "args": CMDARGS,
                    },
                }
            ],
            # "service_account": service_account,
        },
    }
    parent = f"projects/{project}/locations/{region}"
    response = client.create_custom_job(parent=parent, custom_job=custom_job)
    print("response:", response)

    return response


def run_prediction_job(
    model_datetime: str,
    keyword_table: str,
    prediction_table: str,
    product_table: str,
    snippet_table: str,
    brand_table: str,
    firestore_categories: str,
    project_number: int,
    project: str,
    bucket: str,
    region: str,
    api_endpoint: str,
    package_uri: str,
    # service_account: str,
    batch_size: int = 6,
    model_step: int = None,
):
    EVAL_GPU, EVAL_NGPU = (aip.AcceleratorType.NVIDIA_TESLA_P100, 1)
    EVAL_IMAGE = "europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:m75"
    EVAL_COMPUTE = "n1-standard-8"
    TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    JOB_NAME = "predict_sap_classifier_" + TIMESTAMP

    CMDARGS = [
        "--bucket=" + bucket,
        "--keyword-table=" + keyword_table,
        "--product-table=" + product_table,
        "--prediction-table=" + prediction_table,
        "--snippet-table=" + snippet_table,
        "--brand-table=" + brand_table,
        "--model-datetime=" + model_datetime,
        "--batch-size=" + str(batch_size),
        "--project=" + project,
        "--firestore-categories=" + firestore_categories,
        "--project-number=" + str(project_number),
    ]

    if model_step:
        CMDARGS.append("--model-step=" + str(model_step))

    MODEL_DISPLAY_NAME = JOB_NAME

    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    custom_job = {
        "display_name": MODEL_DISPLAY_NAME,
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": EVAL_COMPUTE,
                        "accelerator_type": EVAL_GPU,
                        "accelerator_count": EVAL_NGPU,
                    },
                    "replica_count": 1,
                    "python_package_spec": {
                        "executor_image_uri": EVAL_IMAGE,
                        "package_uris": [package_uri],
                        "python_module": "training.predict",
                        "args": CMDARGS,
                    },
                }
            ],
            # "service_account": service_account,
        },
    }
    parent = f"projects/{project}/locations/{region}"
    response = client.create_custom_job(parent=parent, custom_job=custom_job)
    print("response:", response)

    return response

def train():
    return run_training_job(
        data="gs://haba-ws/data.csv",
        epochs=10,
        project="bence-bial-sandbox",
        bucket="haba-ws",
        output_path="output-debug",
        batch_size=16,
        eval_steps=1000,
        region="europe-west4",
        api_endpoint="europe-west4-aiplatform.googleapis.com",
        package_uri="gs://haba-ws/container/text-classification-0.1.tar.gz"
    )