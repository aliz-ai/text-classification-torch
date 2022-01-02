# Text Classification with Hugginface pretrained models in PyTorch

## Local installation

```sh
conda create -n text-classification python=3.8
conda activate text-classification
#pip install -r requirements.txt
pip install -e .
```

##

```sh
python setup.py sdist --formats=gztar
gsutil cp dist/text-classification-0.1.tar.gz gs://haba-ws/container/
```