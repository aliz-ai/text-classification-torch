from datetime import datetime
from typing import Union

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
    early_stopping_patience: int,
    region: str,
    api_endpoint: str,
    package_uri: str,
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
        "--eval-steps=" + str(eval_steps),
        "--early-stopping-patience=" + str(early_stopping_patience)
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
            ]
        }
    }
    parent = f"projects/{project}/locations/{region}"
    response = client.create_custom_job(parent=parent, custom_job=custom_job)
    print("response:", response)

    return response
