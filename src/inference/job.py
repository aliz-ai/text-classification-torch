from datetime import datetime

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip


def run_prediction_job(
    data: str,
    project: str,
    bucket: str,
    model_path: str,
    label_id_path: str,
    output_path: str,
    batch_size: int,
    region: str,
    api_endpoint: str,
    package_uri: str,
):
    """Creates a parameterized prediction job with a custom model."""

    EVAL_GPU, EVAL_NGPU = (aip.AcceleratorType.NVIDIA_TESLA_V100, 1)
    EVAL_IMAGE = "europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:m82"
    EVAL_COMPUTE = "n1-standard-4"
    TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    JOB_NAME = "predict_text_classifier_" + TIMESTAMP

    CMDARGS = [
        "--bucket=" + bucket,
        "--project=" + project,
        "--data=" + data,
        "--model-path=" + model_path,
        "--label-id-path=" + label_id_path,
        "--output-path=" + output_path,
        "--batch-size=" + str(batch_size),
    ]

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
                        "python_module": "inference.predict",
                        "args": CMDARGS,
                    },
                }
            ]
        },
    }
    parent = f"projects/{project}/locations/{region}"
    response = client.create_custom_job(parent=parent, custom_job=custom_job)
    print("response:", response)

    return response


def run_evaluation_job(
    data: str,
    project: str,
    bucket: str,
    model_path: str,
    label_id_path: str,
    output_path: str,
    batch_size: int,
    region: str,
    api_endpoint: str,
    package_uri: str,
):
    """Creates a parameterized evaluation job with a custom model."""

    EVAL_GPU, EVAL_NGPU = (aip.AcceleratorType.NVIDIA_TESLA_V100, 1)
    EVAL_IMAGE = "europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:m82"
    EVAL_COMPUTE = "n1-standard-4"
    TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    JOB_NAME = "evaluate_text_classifier_" + TIMESTAMP

    CMDARGS = [
        "--bucket=" + bucket,
        "--project=" + project,
        "--data=" + data,
        "--model-path=" + model_path,
        "--label-id-path=" + label_id_path,
        "--output-path=" + output_path,
        "--batch-size=" + str(batch_size),
    ]

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
                        "python_module": "inference.evaluate",
                        "args": CMDARGS,
                    },
                }
            ]
        },
    }
    parent = f"projects/{project}/locations/{region}"
    response = client.create_custom_job(parent=parent, custom_job=custom_job)
    print("response:", response)

    return response
