# sentimentanalysis-COVID19news
Sentiment Analysis of selected news articles pre and during COVID-19 pandemic

This repository provides code and supplementary materials for the paper entitled 
'Large language models for sentiment analysis of newspaper articles during COVID-19: The Guardian'.

Here is the link to our Paper: Rohitash Chandra, Baicheng Zhu, Qingying Fang, Eka Shinjikashvili: https://arxiv.org/abs/2405.13056 

# Preparing Dataset

We used a dataset of 10,000 manually labeled English tweets containing 10 different sentiments for training and testing.
And the SenWave dataset is from GitHub: SenWave: A Fine-Grained Sentiment Analysis Dataset for COVID-19 Tweets. 
https://github.com/gitdevqiang/SenWave?tab=readme-ov-file#senwave-a-fine-grained-sentiment-analysis-dataset-for-covid-19-tweets

After fine-tuning the model, we used it to label emotions in articles from The Guardian
on Kaggle. We selected sections including Australia News, UK News, World News, and Opinion 
for a more detailed analysis.
Note the project uses dataset from Kaggle: Guardian News Articles. Kaggle. 
https://www.kaggle.com/datasets/adityakharosekar2/guardian-news-articles

# Notebooks for project: main run code.
The repository includes individual Jupyter Notebook files for BERT model, RoBERTa model, visualisation and result part, namely:

model part:

BERT_model/BERT_model.ipynb

RoBERTa_model/Roberta_finetune1.0.ipynb

Visualisation part:

Note that the Jupyter Notebook files in the visualization section contains images of our results.

Visualisation/visualization2.ipynb

Visualisation/target_ngrams.ipynb

Visualisation/polarity_scores.py

# Results part

We have article files labelled using two models, which we named BERT and RoBERTa.

Our framework is produced by visio, the following is the URL: https://unsw-my.sharepoint.com/:u:/g/personal/z5427897_ad_unsw_edu_au/EW4Py1_GtdtLhjuj5xHRWjYBWmh_vkWNVqkMFsQkK0wwmw?e=yvtldM


