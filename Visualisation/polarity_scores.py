import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns



data_path = r"./roberta_final1.0.csv"
roberta_data = pd.read_csv(data_path)
roberta_data = roberta_data.dropna()
roberta_data = roberta_data.reset_index().drop(['index'], axis = 1)

print(roberta_data.head())
aus_roberta = roberta_data[(roberta_data['sectionName']=='Australia news')]
uk_roberta = roberta_data[(roberta_data['sectionName']=='UK news')]
world_roberta = roberta_data[roberta_data['sectionName'] == 'World news']


sentiment_polarity = {
    'Optimistic': 2,
    'Thankful': 3,
    'Joking': 1,
    'Pessimistic': -4,
    'Anxious': -2,
    'Sad':  -3,
    'Annoyed': -1,
    'Denial': -5,
    'Empathetic': 0,
    'Official report': 0,
}

# Define the list of sentiment labels and countries
labels = ['Optimistic', 'Thankful', 'Empathetic', 'Pessimistic', 'Anxious', 'Sad', 'Annoyed', 'Denial',
          'Official report', 'Joking']
countries = ['aus_roberta', 'uk_roberta', 'world_roberta']

# Define the weight ratios for sentiment labels
weight_ratios = {'Optimistic': 3, 'Thankful': 2, 'Empathetic': 0, 'Pessimistic': -3, 'Anxious': -2, 'Sad': -2,
                 'Annoyed': -1, 'Denial': -4, 'Official report': 0, 'Joking': 1}


# Function to calculate polarity scores for a given DataFrame
def calculate_polarity_scores(df, weight_ratios):
    # Apply weight ratios to sentiment labels
    weighted_df = df[labels].apply(lambda x: x * weight_ratios[x.name])

    # Sum up the weighted scores for each row
    df.loc[:, 'score'] = weighted_df.sum(axis=1)

    # Normalize the score to make it between -1 and 1
    df.loc[:, 'score'] = df['score'] / len(labels)

    return df


# Calculate polarity scores for each country and store them in a dictionary
polarity_scores = {}
for country in countries:
    df_country = globals()[country].copy()  # Get a copy of the DataFrame by its variable name
    df_country = calculate_polarity_scores(df_country, weight_ratios)
    polarity_scores[country] = df_country['score']

# Create a DataFrame from the dictionary of polarity scores
polarity_df = pd.DataFrame(polarity_scores)

# Count the occurrences of each unique polarity score
polarity_counts = polarity_df.apply(pd.value_counts).fillna(0)

# Plot the distribution of polarity scores for each country on the same graph
plt.figure(figsize=(10, 6))
for country in countries:
    plt.plot(polarity_counts.index, polarity_counts[country], marker='o', label=country)

# Add labels and title
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
# plt.title('Distribution of Polarity Scores for Different Regions')
plt.grid(True)
plt.legend(labels=['Australia', 'UK', 'World'])  # Specify custom labels

# Show the plot
plt.show()

# Define the list of sentiment labels and countries
labels = ['Optimistic', 'Thankful', 'Empathetic', 'Pessimistic', 'Anxious', 'Sad', 'Annoyed', 'Denial',
          'Official report', 'Joking']
countries = ['aus_roberta', 'uk_roberta', 'world_roberta']
country_names = ['Australia', 'UK', 'World']  # Custom country names

# Define the weight ratios for sentiment labels
weight_ratios = {'Optimistic': 3, 'Thankful': 2, 'Empathetic': 0, 'Pessimistic': -3, 'Anxious': -2, 'Sad': -2,
                 'Annoyed': -1, 'Denial': -4, 'Official report': 0, 'Joking': 1}


# Function to calculate polarity scores for a given DataFrame
def calculate_polarity_scores(df, weight_ratios):
    # Apply weight ratios to sentiment labels
    weighted_df = df[labels].apply(lambda x: x * weight_ratios[x.name])

    # Sum up the weighted scores for each row
    df.loc[:, 'score'] = weighted_df.sum(axis=1)

    # Normalize the score to make it between -1 and 1
    df.loc[:, 'score'] = df['score'] / len(labels)

    return df


# Calculate polarity scores for each country and store them in a dictionary
polarity_scores = {}
for country in countries:
    df_country = globals()[country].copy()  # Get a copy of the DataFrame by its variable name
    df_country = calculate_polarity_scores(df_country, weight_ratios)
    polarity_scores[country] = df_country['score']

# Create a DataFrame from the dictionary of polarity scores
polarity_df = pd.DataFrame(polarity_scores)

# Count the occurrences of each unique polarity score
polarity_counts = polarity_df.apply(pd.value_counts).fillna(0)

# Plot the distribution of polarity scores for each country on the same graph
plt.figure(figsize=(10, 6))

# Plot boxplot for each country
sns.boxplot(data=polarity_df, notch=True, showfliers=False)

# Add labels and title
plt.xlabel('Region')
plt.ylabel('Polarity Score')
plt.title('Distribution of Polarity Scores for Different Regions')

# Set custom x-axis labels
plt.xticks(range(len(countries)), country_names)

# Show the plot
plt.show()


def calculate_polarity_scores(df):
    # Define the weight ratios for sentiment labels
    weight_ratios = {'Optimistic': 3, 'Thankful': 2, 'Empathetic': 0, 'Pessimistic': -3, 'Anxious': -2, 'Sad': -2,
                     'Annoyed': -1, 'Denial': -4, 'Official report': 0, 'Joking': 1}
    # Apply weight ratios to sentiment labels
    weighted_df = df[labels].apply(lambda x: x * weight_ratios[x.name])

    # Sum up the weighted scores for each row
    df.loc[:, 'score'] = weighted_df.sum(axis=1)

    # Normalize the score to make it between -1 and 1
    df.loc[:, 'score'] = df['score'] / len(labels)

    return df
def plot_sentiment_over_quarters_multi(df1, df2, df3, region1, region2, region3, start_date='2018-01-01', end_date='2022-03-31'):

    df1 = df1.copy()
    df2 = df2.copy()
    df3 = df3.copy()

    df1 = calculate_polarity_scores(df1)
    df2 = calculate_polarity_scores(df2)
    df3 = calculate_polarity_scores(df3)

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df3 = df3.reset_index(drop=True)

    df1['webPublicationDate'] = pd.to_datetime(df1['webPublicationDate'])
    df2['webPublicationDate'] = pd.to_datetime(df2['webPublicationDate'])
    df3['webPublicationDate'] = pd.to_datetime(df3['webPublicationDate'])

    df1.set_index('webPublicationDate', inplace=True)
    df2.set_index('webPublicationDate', inplace=True)
    df3.set_index('webPublicationDate', inplace=True)

    df1_resampled = df1.resample('Q').mean(numeric_only=True)
    df2_resampled = df2.resample('Q').mean(numeric_only=True)
    df3_resampled = df3.resample('Q').mean(numeric_only=True)

    quarters = pd.date_range(start=start_date, end=end_date, freq='Q')

    df_resampled_filled = pd.DataFrame(index=quarters)

    df_resampled_filled[f'{region1}_Sentiment'] = df1_resampled['score'].fillna(0)
    df_resampled_filled[f'{region2}_Sentiment'] = df2_resampled['score'].fillna(0)
    df_resampled_filled[f'{region3}_Sentiment'] = df3_resampled['score'].fillna(0)

    plt.figure(figsize=(10, 6))
    plt.plot(df_resampled_filled.index, df_resampled_filled[f'{region1}_Sentiment'], label=region1)
    plt.plot(df_resampled_filled.index, df_resampled_filled[f'{region2}_Sentiment'], label=region2)
    plt.plot(df_resampled_filled.index, df_resampled_filled[f'{region3}_Sentiment'], label=region3)
    # plt.title('Sentiment Polarity over Quarters')
    # plt.xlabel('Quarter')
    plt.ylabel('Sentiment Polarity')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()


plot_sentiment_over_quarters_multi(aus_roberta, uk_roberta, world_roberta, 'Australia', 'UK', 'World')
