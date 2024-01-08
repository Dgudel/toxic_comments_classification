import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

import pytorch_lightning as pl

import transformers
from transformers import (
    AutoTokenizer,
    DistilBertTokenizer,
    DistilBertForMaskedLM,
    DistilBertModel,
    DistilBertForSequenceClassification,
    DistilBertConfig,
)

from tokenizers import trainers, models, Tokenizer, pre_tokenizers
from transformers import pipeline
from sklearn.model_selection import train_test_split


class WikipediaCommentDataset(Dataset):
    """
    Dataset class for loading comments data for text classification using DistilBERT.
    """
    def __init__(self, df: pd.DataFrame, max_sequence_length: int = 100, labels_available: bool = True):
        """
        Initializes the CommentsDataset class.

        Args:
            df (pd.DataFrame): Dataframe containing the comments data.
            max_sequence_length (int, optional): Maximum sequence length for tokenization. Defaults to 100.
            labels_available (bool, optional): Whether labels are available in the dataset. Defaults to True.
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
        self.df = df
        self.max_sequence_length = max_sequence_length
        self.labels_available = labels_available

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Gets an item from the dataset at the given index.

        Args:
            idx (int): Index of the item to be retrieved.

        Returns:
            dict: Encoded input data and labels.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        encoded = self.tokenizer.encode_plus(
            text=self.df.loc[idx, "comment_text"],
            add_special_tokens=True,
            max_length=self.max_sequence_length,
            padding="max_length",
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}

        if self.labels_available:
            labels = self.df.iloc[idx, 2:8].values.astype(np.float32)
            encoded["labels"] = torch.tensor(labels, dtype=torch.float)

        return encoded



