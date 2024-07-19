from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
# from google.colab import drive
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import nltk
import csv
import re
import demoji
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import operator
import spacy
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator
from sklearn.metrics import hamming_loss, jaccard_score, label_ranking_average_precision_score, f1_score
from sklearn.model_selection import train_test_split

from word_embeddings import load_embeddings, build_vocab, check_coverage

SEED = 1024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler
import torchtext
from torchtext import data
# Mount Google Drive
# drive.mount('/content/drive')
# Load the RoBERTa tokenizer
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
# model.to(device)
# Read the CSV file into a DataFrame
# file_path = "/content/drive/MyDrive/Colab Notebooks/labeledEn.csv"
# load senwave
#file_path = 'D:/UNSW/2024T1/MATH5925/code/labeledEn.csv'
file_path = 'labeledEn.csv'

senwave = pd.read_csv(file_path)
# Check the shape of data
print(senwave.shape)
# Print the first few rows of the DataFrame
# print(senwave.head())

from preprocessing.preprocess import (preprocess, contractions,
                                      contractionsWithAnotherInvertedComma)

pp_class = preprocess(senwave, contractions, contractionsWithAnotherInvertedComma)
senwave['Tweet'] = senwave['Tweet'].apply(lambda x : pp_class.preprocess_tweet(x))

senwave['Tweet'] = senwave['Tweet'].str.lower()
pd.set_option('display.max_columns', None)
print(senwave.head(10))

# for roberta it is not need
# Load the pre-trained GloVe word embeddings model file from the specified path and store it as a dictionary-formatted embedding matrix.
# from word_embeddings import load_embeddings, build_vocab, check_coverage
# GLOVE_EMBEDDING_FILE = 'D:/UNSW/2024T1/MATH5925/code/glove.840B.300d.txt'
GLOVE_EMBEDDING_FILE = 'glove.840B.300d.txt'
glove_embeddings = load_embeddings(GLOVE_EMBEDDING_FILE)
# Print the number of loaded word vectors.
print(f'loaded {len(glove_embeddings)} word vectors ')
vocab = build_vocab(list(senwave['Tweet'].apply(lambda x : x.split())))
oov = check_coverage(vocab, glove_embeddings)
oov[:10]

from preprocessing.wordReplace import bruteGen
senwave['Tweet'] = senwave['Tweet'].apply(lambda x : bruteGen(x))

sen_train, sen_test = train_test_split(senwave, train_size = 0.9, random_state = 1024)

sen_train.to_csv("train.csv", index = False)
sen_test.to_csv("test.csv", index = False)
print(sen_train.head(1))


# not suit for roberta
nlp = spacy.load("en_core_web_sm")
def tokenizer(tweet):
    tweet = re.sub(r'[\n]', ' ', tweet)
    return [tok.text for tok in nlp(tweet)]

tokenizer = AutoTokenizer.from_pretrained("roberta-base")# Use RoBERTa tokenizer
# Define a function to tokenize a text using RoBERTa tokenizer
def tokenize_with_roberta(tweet):
    # Replace newline characters with space
    tweet = re.sub(r'[\n]', ' ', tweet)
    # Tokenize the text using RoBERTa tokenizer
    return tokenizer.tokenize(tweet)

TWEET = Field(sequential=True, tokenize=tokenize_with_roberta)
LABEL = Field(sequential=False, use_vocab=False)

dataFields = [("ID", None), ("Tweet", TWEET), ("Optimistic", LABEL), ("Thankful", LABEL),
              ("Empathetic", LABEL), ("Pessimistic", LABEL), ("Anxious", LABEL), ("Sad", LABEL),
              ("Annoyed", LABEL), ("Denial", LABEL), ("Official report", LABEL), ("Joking", LABEL)]

'''train_dataset, test_dataset = TabularDataset.splits(
    path='D:/UNSW/2024T1/MATH5925/code', train='train.csv', test='test.csv', format='csv', fields=dataFields, skip_header=True
)'''
train_dataset, test_dataset = TabularDataset.splits(
    path='.', train='train.csv', test='test.csv', format='csv', fields=dataFields, skip_header=True
)

print("Number of training samples : {}\n Number of testing samples : {}".format(len(train_dataset), len(test_dataset)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# TWEET.build_vocab(train_dataset, vectors = 'glove.840B.300d')
# Drop the 'ID' column from the DataFrame
df = senwave.drop(['ID'], axis=1)
# Create a new column named 'list' containing values of columns 1 to 10 as lists
df['list'] = df[df.columns[1:11]].values.tolist()

# Create a new DataFrame 'new_df' containing the 'Tweet' column and the newly created 'list' column
new_df = df[['Tweet', 'list']].copy()
new_df.head()

MAX_LEN = 200  # Based on the length of tweets
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')# Use RoBERTa tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.tweet = dataframe['Tweet']
        self.targets = self.dataframe.list
        self.max_len = max_len

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, index):
        tweet = str(self.tweet[index])
        tweet = " ".join(tweet.split())

        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,  # Add special tokens for RoBERTa
            max_length=self.max_len,
            padding='max_length',  # Pad to max_length
            return_token_type_ids=True,
            return_tensors='pt',  # Return PyTorch tensors
            truncation=True  # Truncate sequences longer than max_length
        )
        input_ids = inputs['input_ids'].squeeze(0)  # Remove the added batch dimension
        attention_mask = inputs['attention_mask'].squeeze(0)  # Remove the added batch dimension

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# For the training dataset
train_dataset = sen_train.drop(['ID'], axis=1)  # Drop the 'ID' column from the training data
num_columns = len(train_dataset.columns)
train_dataset['list'] = train_dataset[train_dataset.columns[1:num_columns]].values.tolist()

# Check the length of the feature vectors in the 'list' column
# Assume the length of the feature vectors should be 10
feature_vector_length = 10
train_dataset['list'] = train_dataset['list'].apply(lambda x: x[:feature_vector_length])

train_df = train_dataset[['Tweet', 'list']].copy()  # Select only the 'Tweet' and 'list' columns
train_df = train_df.reset_index(drop=True)  # Reset the index of the DataFrame, dropping the old index

# For the testing dataset
test_dataset = sen_test.drop(['ID'], axis=1)
num_columns = len(test_dataset.columns)
test_dataset['list'] = test_dataset[test_dataset.columns[1:num_columns]].values.tolist()

# Check the length of the feature vectors in the 'list' column
test_dataset['list'] = test_dataset['list'].apply(lambda x: x[:feature_vector_length])

test_df = test_dataset[['Tweet', 'list']].copy()
test_df = test_df.reset_index(drop=True)  # Reset the index of the DataFrame, dropping the old index

training_set = CustomDataset(train_df, tokenizer, MAX_LEN)

testing_set = CustomDataset(test_df, tokenizer, MAX_LEN)

# Define parameters for training DataLoader
train_params = {'batch_size': TRAIN_BATCH_SIZE,  # Set batch size
                'shuffle': True,  # Shuffle the data
                'num_workers': 0}

# Define parameters for testing DataLoader
test_params = {'batch_size': VALID_BATCH_SIZE,  # Set batch size
                'shuffle': True,  # Shuffle the data
                'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# Creating the Transformer model
class RoBERTaCustom(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

model = RoBERTaCustom(num_classes=10)
model.to(device)

# Define the loss function
def loss_fn(outputs, targets):
    # Use Binary Cross Entropy with Logits Loss for calculating the loss
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# Define the optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)



def train(epoch):
    # Set the model to training mode
    model.train()
    # Initialize total loss for this epoch
    total_loss = 0

    # Iterate over the training data loader

    for batch_idx, data in enumerate(tqdm(training_loader), 0):
        # Move input tensors to the appropriate device (GPU or CPU)
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['targets'].to(device)

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Print the shapes of outputs and targets for debugging
        # print("Output shape:", outputs.shape)
        # print("Target shape:", targets.shape)

        # Calculate the batch loss
        loss = loss_fn(outputs, targets)

        # Accumulate the total loss for this epoch
        total_loss += loss.item()

        # Print training progress every 2000 batches
        if batch_idx % 2000 == 0:
            print(f'Iteration: {batch_idx+1}, Epoch: {epoch+1}, Loss: {total_loss/(batch_idx+1)}')

        # Zero the gradients, backward pass, and update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

EPOCHS = 4
for epoch in range(EPOCHS):
    train(epoch)


def valid():
    model.eval()
    req_targets = []
    req_outputs = []
    valid_loss = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(testing_loader, 0):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['targets'].to(device)

            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate the batch loss
            loss = loss_fn(outputs, targets)
            valid_loss += loss.item()

            # Append targets and outputs
            req_targets.extend(targets.cpu().detach().numpy().tolist())
            req_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    # Calculate the average validation loss
    valid_loss /= len(testing_loader)

    return req_outputs, req_targets, valid_loss

outputs, targets, valid_loss = valid()
outputs = np.array(outputs)
targets = np.array(targets)
int_outputs = np.zeros_like(outputs)

for i in range(outputs.shape[0]):
    for j in range(outputs.shape[1]):
        if outputs[i][j] >= 0.5:
            int_outputs[i][j] = 1

bert_ham_loss = hamming_loss(targets, int_outputs)
bert_jacc_score = jaccard_score(targets, int_outputs, average='samples')
bert_lrap = label_ranking_average_precision_score(targets, outputs)
bert_f1_macro = f1_score(targets, int_outputs, average='macro')
bert_f1_micro = f1_score(targets, int_outputs, average='micro')

print("Test Loss:", valid_loss)
print("Hamming Loss:", bert_ham_loss)
print("Jaccard Score:", bert_jacc_score)
print("Label Ranking Average Precision Score:", bert_lrap)
print("F1 Macro Score:", bert_f1_macro)
print("F1 Micro Score:", bert_f1_micro)

# torch.save(model, f = 'D:/UNSW/2024T1/MATH5925/code/roberta-finetuned3.pth')
torch.save(model, 'roberta-finetuned.pth')
print('All files saved')