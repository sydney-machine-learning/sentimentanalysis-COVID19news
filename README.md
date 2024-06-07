# Sentiment Analysis of COVID-19 News Articles

This repository provides code and supplementary materials for the paper titled 'Large language models for sentiment analysis of newspaper articles during COVID-19: The Guardian'.

**Paper Link:** [Rohitash Chandra, Baicheng Zhu, Qingying Fang, Eka Shinjikashvili](https://arxiv.org/abs/2405.13056)

**Seminar Recording:** [YouTube](https://www.youtube.com/watch?v=TU6Vvoj4U5Y&ab_channel=transitional-ai)

Our framework is produced by visio, and the URL is: [Visio Framework](https://unsw-my.sharepoint.com/:u:/g/personal/z5427897_ad_unsw_edu_au/EW4Py1_GtdtLhjuj5xHRWjYBWmh_vkWNVqkMFsQkK0wwmw?e=yvtldM)

## Preparing Dataset

We used a dataset of 10,000 manually labeled English tweets containing 10 different sentiments for training and testing. Additionally, the SenWave dataset from GitHub was utilised: 
[SenWave Dataset](https://github.com/gitdevqiang/SenWave?tab=readme-ov-file#senwave-a-fine-grained-sentiment-analysis-dataset-for-covid-19-tweets)

After fine-tuning the model, we used it to label sentiments in articles from The Guardian on Kaggle. Sections including Australia News, UK News, World News, and Opinion were selected 
for a detailed analysis. It's worth noting that the project also uses the Guardian News Articles dataset from Kaggle: 
[Guardian News Articles Dataset](https://www.kaggle.com/datasets/adityakharosekar2/guardian-news-articles)

### Load Dataset using Google Drive
Note that the following code demonstration is mainly applied to the RoBERTa model.
To load the dataset in Google Colab, follow these steps:

1. **Mount Google Drive**: Use the command `drive.mount('/content/drive')` in your notebook to mount Google Drive.
2. **Load Dataset**: Utilize the `pd.read_csv()` function to read the CSV file. Replace the `file_path` variable with your CSV file path.

```python
# Mount Google Drive
drive.mount('/content/drive')
# Read the CSV file into a DataFrame
file_path = "/content/drive/MyDrive/Colab Notebooks/labeledEn.csv"
df = pd.read_csv(file_path)
```
### Saving DataFrames as CSV Files
Note that the following code demonstration is mainly applied to the RoBERTa model.
To save a Pandas DataFrame as a CSV file, you can use the `to_csv()` function. Here's how you can do it:

```python
# Assuming `sen_train` and `sen_test` are your Pandas DataFrames for the training and testing sets
sen_train.to_csv("train.csv", index=False)
sen_test.to_csv("test.csv", index=False)
```


## Notebooks for project: main run code.
The repository includes individual Jupyter Notebook files for BERT model, RoBERTa model, visualisation and result part, namely:

model part:

BERT_model/BERT_model.ipynb

RoBERTa_model/Roberta_finetune1.0.ipynb

### Saving a PyTorch Model

To save a PyTorch model, you can use the `torch.save()` function. Here's how you can do it:
Note that the following code demonstration is mainly applied to the RoBERTa model.
```python
import torch

# Assuming `model` is your PyTorch model
model = ...

# File path to save the model
file_path = '/content/drive/MyDrive/RoBERTa_ft.pth'

# Save the model
torch.save(model, f=file_path)
```

## Visualisation part:

Note that the Jupyter Notebook files in the visualization section contains images of our results.

Visualisation/visualization2.ipynb

Visualisation/target_ngrams.ipynb

Visualisation/polarity_scores.py

## Results part

We have article files labelled using two models, which we named BERT and RoBERTa.
