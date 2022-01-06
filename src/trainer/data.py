from typing import List

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """
    Torch dataset for text data. Samples are returned in the required format for the
    Huggingface Trainer. Text is tokenized online.
    """

    def __init__(self, texts: List[str], labels: List[int], bert_version: str):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_version)

        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        sample = {
            **self.tokenizer(text, padding="max_length", truncation=True),
            "label": self.labels[idx],
            "text": text,
        }
        return sample
