from setuptools import find_packages, setup

DESCRIPTION = "Text Classification with Huggingface"

requirements = [
    "pandas==1.3.5",
    "transformers==4.14.1",
    "scikit-learn==1.0.2",
    "numpy==1.22.0",
    "fsspec==2021.11.1",
    "gcsfs==2021.11.1",
    "google-cloud-storage==1.43.0",
    "google-cloud-logging==2.7.0",
    "datasets==1.17.0",
    "tensorboardX==2.4.1",
    "tensorboard==2.7.0",
    "google-cloud-aiplatform==1.8.1",
]

setup(
    name="text-classification",
    version="0.1",
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author_email="dev@aliz.ai",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.py"]},
    install_requires=requirements,
)
