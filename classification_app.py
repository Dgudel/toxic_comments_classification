# +
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import optim

from flask import Flask, request, jsonify, render_template, render_template_string
import transformers
from transformers import (
    AutoTokenizer,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertConfig,
    pipeline,
)
import re
import os
import random

from wordcloud import WordCloud, STOPWORDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from io import BytesIO
import base64

from CommentsDataset import WikipediaCommentDataset
from CommentsModel import CommentsModel
# -

app = Flask(__name__)

# +
checkpoint = "multilabel_classification_5_epochs_finetuned.ckpt"
model = CommentsModel.load_from_checkpoint(checkpoint, num_classes=6)
device = torch.device("mps")
model = model.eval().to(device)

test = pd.read_csv("data/test.csv")
test_dataset = WikipediaCommentDataset(test, labels_available=False)
BATCH_SIZE = 5
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)

label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
num_batches = 30
num_samples = num_batches * test_loader.batch_size
sampler = SubsetRandomSampler(indices=torch.randperm(len(test_loader.dataset))[:num_samples])

random_loader = DataLoader(
test_loader.dataset, sampler=sampler, batch_size=test_loader.batch_size)

# -

test.head()


def clean_text(text: str) -> str:
    """
    Cleans the given text by removing extra spaces and special characters.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = " ".join(text.split())
    text = re.sub("[^A-Za-z0-9,.!? ]+", "", text)
    return text


test['comment_text'] = test['comment_text'].apply(clean_text)


def generate_wordcloud_image(data: str, filename: str = "wordcloud.png") -> str:
    """Generate a word cloud image from the given data and save it to a file.

    Parameters:
        data (str): The data to be used for creating the word cloud.
        filename (str): The name of the image file to save the word cloud in.

    Returns:
        str: The path to the saved image file.
    """
    wordcloud = WordCloud(
        background_color="white",
        stopwords=STOPWORDS,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1,
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis("off")
    plt.imshow(wordcloud, interpolation="bilinear")
    image_path = os.path.join("static", filename)  
    fig.savefig(image_path)
    plt.close()  
    return image_path


def wordcloud_to_base64(data: str) -> str:
    wordcloud = WordCloud(
        background_color="white",
        stopwords=STOPWORDS,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1,
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis("off")
    plt.imshow(wordcloud)
    plt.tight_layout()

    img = BytesIO()
    fig.savefig(img, format="png")
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode()

    plt.close()

    return img_b64


def create_dataframe_row(comment_text):

    random_id = random.randint(1, 1e6)
    df = pd.DataFrame({
        'id': [random_id],
        'comment_text': [comment_text]
    })

    return df


@app.route('/')
def home():       
        
    all_preds = []
    all_indices = []

    with torch.no_grad():
        for i, batch in enumerate(random_loader):
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask"]
            }

            outputs = model(**batch)
            sigmoids = torch.sigmoid(outputs)
            preds = (sigmoids > 0.5).to(torch.int)

            all_preds.append(preds.cpu())

            start_idx = i * random_loader.batch_size
            end_idx = start_idx + random_loader.batch_size
            batch_indices = sampler.indices[start_idx:end_idx]
            all_indices.extend(batch_indices)

    all_preds = torch.cat(all_preds, 0)
    all_indices = [index.item() for index in all_indices]
    all_preds_list = all_preds.tolist()
    
    selected_df = test.iloc[all_indices].copy()
    preds_df = pd.DataFrame(all_preds_list, columns=label_names)
    selected_df.reset_index(drop=True, inplace=True)
    predicted_df = pd.concat([selected_df, preds_df], axis=1)
    
    selected_comments = predicted_df["comment_text"].tolist()

    image_path = generate_wordcloud_image(selected_comments)

    return render_template("wordcloud.html", image_url=image_path)

@app.route('/classify_wordcloud', methods=['POST', 'GET'])
def classify_wordcloud():
     
    all_preds = []
    all_indices = []

    with torch.no_grad():
        for i, batch in enumerate(random_loader):
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask"]
            }

            outputs = model(**batch)
            sigmoids = torch.sigmoid(outputs)
            preds = (sigmoids > 0.5).to(torch.int)

            all_preds.append(preds.cpu())

            start_idx = i * random_loader.batch_size
            end_idx = start_idx + random_loader.batch_size
            batch_indices = sampler.indices[start_idx:end_idx]
            all_indices.extend(batch_indices)

    all_preds = torch.cat(all_preds, 0)
    all_indices = [index.item() for index in all_indices]
    all_preds_list = all_preds.tolist()

    selected_df = test.iloc[all_indices].copy()
    preds_df = pd.DataFrame(all_preds_list, columns=label_names)
    selected_df.reset_index(drop=True, inplace=True)
    predicted_df = pd.concat([selected_df, preds_df], axis=1)

    predicted_inapr = predicted_df[
    (predicted_df["toxic"] == 1)
    | (predicted_df["severe_toxic"] == 1)
    | (predicted_df["obscene"] == 1)
    | (predicted_df["threat"] == 1)
    | (predicted_df["insult"] == 1)
    | (predicted_df["identity_hate"] == 1)]
    

    predicted_comments = predicted_inapr["comment_text"].tolist()
    
    # Generate word cloud
    image_b64 = wordcloud_to_base64(' '.join(predicted_comments))

    return render_template('wordcloud_inappropriate.html', image_b64=image_b64)


@app.route('/classify', methods=['GET', 'POST'])
def classify():
     
    all_preds = []
    all_indices = []

    with torch.no_grad():
        for i, batch in enumerate(random_loader):
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask"]
            }

            outputs = model(**batch)
            sigmoids = torch.sigmoid(outputs)
            preds = (sigmoids > 0.5).to(torch.int)

            all_preds.append(preds.cpu())

            start_idx = i * random_loader.batch_size
            end_idx = start_idx + random_loader.batch_size
            batch_indices = sampler.indices[start_idx:end_idx]
            all_indices.extend(batch_indices)

    all_preds = torch.cat(all_preds, 0)
    all_indices = [index.item() for index in all_indices]
    all_preds_list = all_preds.tolist()

    selected_df = test.iloc[all_indices].copy()
    preds_df = pd.DataFrame(all_preds_list, columns=label_names)
    selected_df.reset_index(drop=True, inplace=True)
    predicted_df = pd.concat([selected_df, preds_df], axis=1)

    predicted_inapr = predicted_df[
    (predicted_df["toxic"] == 1)
    | (predicted_df["severe_toxic"] == 1)
    | (predicted_df["obscene"] == 1)
    | (predicted_df["threat"] == 1)
    | (predicted_df["insult"] == 1)
    | (predicted_df["identity_hate"] == 1)]
    
    predicted_comments = predicted_inapr["comment_text"].tolist()
    
    return render_template('classify.html', comments=predicted_comments)


@app.route('/classify_comment', methods=['POST', 'GET'])
def classify_comment():
    if request.method == 'GET':
        return render_template('comment_input.html')

    elif request.method == 'POST':
        user_comment = request.form['user_comment']
        new_comment = create_dataframe_row(user_comment)

        inputs = WikipediaCommentDataset(new_comment, labels_available=False)
        input_loader = DataLoader(inputs, batch_size=1, shuffle=False, num_workers=3)
        with torch.no_grad():
            for i, batch in enumerate(input_loader):
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in ["input_ids", "attention_mask"]
                }
                outputs = model(**batch)
                sigmoids = torch.sigmoid(outputs)
                preds = (sigmoids > 0.5).to(torch.int).to("cpu")

        preds_np = preds.numpy()
        predicted_df = pd.DataFrame(preds_np, columns=label_names)

        labels_df = predicted_df.apply(lambda x: [x.name if val else None for val in x])
        predicted_labels = labels_df.apply(lambda x: [val for val in x if val is not None], axis=1)

        output =predicted_labels.iloc[0]
        if output:
            message = f"The comment has been detected as: {', '.join(output)}."
        else:
            message = "The comment has not been detected as inappropriate."
        return render_template('comment_classification_output.html', message=message)



if __name__ == '__main__':
    app.run(debug=True)



