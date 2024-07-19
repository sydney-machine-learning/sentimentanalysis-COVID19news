import os

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from roberta_new import CustomDataset, tokenizer, MAX_LEN, device

'''roberta = torch.load(r"D:\UNSW\2024T1\MATH5925\code\roberta-finetuned.pth")
print(roberta)

fdf = pd.read_csv(r"D:\UNSW\2024T1\MATH5925\code\filtered_final1.0.csv")
pd.set_option('display.max_columns', None)
print(fdf.shape)'''

# Confirm the current working directory
current_working_directory = os.getcwd()
print(f"Current Working Directory: {current_working_directory}")

# Load the model using a relative path
roberta = torch.load('roberta-finetuned.pth')
print(roberta)

# Read the CSV file using a relative path
fdf = pd.read_csv('filtered_final1.0.csv')
pd.set_option('display.max_columns', None)
print(fdf.shape)

roberta_df = pd.DataFrame()
roberta_df['Tweet'] = fdf['bodyContent']
values = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 38133
roberta_df['list'] = values
print(roberta_df.head())

test_dataset = CustomDataset(roberta_df, tokenizer, MAX_LEN)

roberta_test_params = {'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 0
                    }

test_loader = DataLoader(test_dataset, **roberta_test_params)


def test():
    roberta.eval()
    roberta_outputs = []

    with torch.no_grad():
        for unw, data in enumerate(tqdm(test_loader), 0):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = roberta(input_ids=input_ids, attention_mask=attention_mask)

            roberta_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return roberta_outputs

test_outputs = test()

test_outputs = np.array(test_outputs)

for i in range(test_outputs.shape[0]):
    for j in range(test_outputs.shape[1]):
        if test_outputs[i][j] >= 0.5: test_outputs[i][j] = 1
        else: test_outputs[i][j] = 0

roberta_df['Optimistic'] = "None"
roberta_df['Thankful'] = "None"
roberta_df['Empathetic'] = "None"
roberta_df['Pessimistic'] = "None"
roberta_df['Anxious'] = "None"
roberta_df['Sad'] = "None"
roberta_df['Annoyed'] = "None"
roberta_df['Denial'] = "None"
roberta_df['Official report'] = "None"
roberta_df['Joking'] = "None"
# roberta_df = roberta_df.drop(['list'], axis=1)

for i in range(len(test_outputs)):
    roberta_df.at[i, 'Optimistic'] = test_outputs[i][0]
    roberta_df.at[i, 'Thankful'] = test_outputs[i][1]
    roberta_df.at[i, 'Empathetic'] = test_outputs[i][2]
    roberta_df.at[i, 'Pessimistic'] = test_outputs[i][3]
    roberta_df.at[i, 'Anxious'] = test_outputs[i][4]
    roberta_df.at[i, 'Sad'] = test_outputs[i][5]
    roberta_df.at[i, 'Annoyed'] = test_outputs[i][6]
    roberta_df.at[i, 'Denial'] = test_outputs[i][7]
    roberta_df.at[i, 'Official report'] = test_outputs[i][8]
    roberta_df.at[i, 'Joking'] = test_outputs[i][9]

roberta_df.head()
# roberta_df.to_csv(r"D:\UNSW\2024T1\MATH5925\code\roberta_final.csv", index=False)
# Save the DataFrame to a CSV file using a relative path
roberta_df.to_csv('roberta_final.csv', index=False)
