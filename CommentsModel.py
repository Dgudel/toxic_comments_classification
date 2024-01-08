import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight


import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import optim
from torchsummary import summary

import pytorch_lightning as pl

import transformers
from transformers import (
    AutoTokenizer,
    DistilBertTokenizer,
    DistilBertForMaskedLM,
    DistilBertModel,
    DistilBertForSequenceClassification,
    DistilBertConfig,
    pipeline,
)

import re
import pickle

from tokenizers import trainers, models, Tokenizer, pre_tokenizers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords
import gensim
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

from collections import defaultdict
from collections import Counter
from typing import Any, List, Tuple, Union

import warnings

warnings.filterwarnings("ignore")

from CommentsDataset import WikipediaCommentDataset


class CommentsModel(pl.LightningModule):
    """
    PyTorch Lightning module for training a DistilBERT model for multi-label classification
    on comments data.

    Attributes:
        num_classes (int): Number of classes for multi-label classification.
        bert (DistilBertForSequenceClassification): Pretrained DistilBERT model.
        lr (float): Learning rate for optimizer.
        sample_weights (torch.Tensor): Sample weights for weighted loss function.
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
    """

    def __init__(self, num_classes: int = 6):
        """
        Initializes the CommentsModel class.

        Args:
            num_classes (int): Number of classes for multi-label classification.
        """
        super().__init__()
        config = DistilBertConfig(
            num_labels=num_classes, problem_type="multi_label_classification"
        )
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", config=config
        )
        self.bert_grad(False)
        self.lr = 1e-3
        self.sample_weights = None
        self.train_losses = []
        self.val_losses = []

    def bert_grad(self, requires_grad: bool):
        """
        Freezes or unfreezes the parameters of the base DistilBERT model for fine-tuning.

        Args:
            requires_grad (bool): Whether or not to require gradients for the parameters.
        """
        for param in self.bert.base_model.parameters():
            param.requires_grad = requires_grad

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Performs a forward pass through the DistilBERT model.

        Args:
            input_ids (torch.Tensor): Tensor of input IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.
            labels (torch.Tensor, optional): Tensor of labels. If provided, the model returns the loss.

        Returns:
            If labels are provided (training/validation):
                loss (torch.Tensor): Loss value.
                logits (torch.Tensor): Logits output from the model.
            Else (inference):
                logits (torch.Tensor): Logits output from the model.
        """
        if labels is not None:
            outputs = self.bert(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss, logits = outputs.loss, outputs.logits
            return loss, logits
        else:
            logits = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
            return logits

    def configure_optimizers(self):
        """
        Configures the optimizer for training the model.

        Returns:
            optimizer (optim.AdamW): AdamW optimizer with specified learning rate and epsilon.
        """
        return optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08
        )

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        Returns:
            loader (DataLoader): DataLoader for the training dataset.
        """
        dataset = WikipediaCommentDataset(df_train)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
        return loader

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            loader (DataLoader): DataLoader for the validation dataset.
        """
        dataset = WikipediaCommentDataset(df_val)
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3
        )
        return loader

    def test_dataloader(self):
        """
        Returns a DataLoader for the test dataset.

        Returns:
            loader (DataLoader): DataLoader for the test dataset.
        """
        dataset = WikipediaCommentDataset(test)
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3
        )
        return loader

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (torch.Tensor): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            weighted_loss (torch.Tensor): Weighted loss value.
        """
        bert_input = batch
        loss, _ = self(**bert_input)
        weighted_loss = loss * self.sample_weights
        loss = loss.mean()
        weighted_loss = weighted_loss.mean()
        self.log("train_loss", loss)
        self.train_losses.append(loss)
        return weighted_loss

    def validation_step(self, val_batch, batch_idx):
        """
        Performs a single validation step.

        Args:
            val_batch (torch.Tensor): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            loss (torch.Tensor): Loss value.
        """
        bert_input = val_batch
        loss, logits = self(**bert_input)
        loss = loss.mean()
        self.log("val_loss", loss)
        self.val_losses.append(loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        """
        Performs a single testing step.

        Args:
            test_batch (torch.Tensor): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            logits (torch.Tensor): Logits output from the model.
        """
        bert_input = test_batch
        logits = self(**bert_input)
        return logits

    def test_epoch_end(self, outputs):
        """
        Performs operations on the outputs of the test data.

        Args:
            outputs (list): List of outputs from the test data.

        Returns:
            all_logits (torch.Tensor): Concatenated logits from all batches.
        """
        all_logits = torch.cat(outputs, dim=0)
        return all_logits

    def on_train_epoch_end(self):
        """
        Performs operations at the end of each training epoch.
        """
        train_loss = torch.stack(self.train_losses).mean()
        self.log("avg_train_loss", train_loss)

    def on_validation_epoch_end(self):
        """
        Performs operations at the end of each validation epoch.
        """
        val_loss = torch.stack(self.val_losses).mean()
        self.log("avg_val_loss", val_loss)


