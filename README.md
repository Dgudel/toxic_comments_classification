# Toxic Comments Classification

The project uses the Wikipedia commment classification dataset uploaded from the Kaggle website (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The dataset includes two csv files:

- train.csv with 159571 entries, each containing columns of Wikipedia comments and columns with info if the comments are labeled as toxic, severe_toxic, obscene, threat, insult, or identity_hate (the labels are one-hot encoded; 0 represents the absence of the characteristic refered by the label and 1 represents the presence).
- test.csv with 153164 entries, each containg comments, but missing the labels; this file will be used for testing.

The purpose of the project is to apply natural language processing and techniques to train multi-label classification models to automatically analyze all messages users write and flag toxic users/comments.

The project consists of two parts:

- the exploratory analysis of the dataset of Wikipedia comments and
- building, training, fine-tuning and evaluating classifiers able to predict comment labels (to identify comments which could be characterized as inapropriate.

The code and outputs of the exploratory analysis and modelling are presented in the file 'multi_label_classification.ipynb'. The class WikipediaCommentDataset, used to tokenize the data, is stored in the separate file 'CommentsDataset.py'.

Trained and fine-tuned models were not uploaded due to limited storage in the Github platform.

Also, the code for the text classification app (text_classification_app.py and html files supposed to be contained in the folder 'templates') is added to the project.

