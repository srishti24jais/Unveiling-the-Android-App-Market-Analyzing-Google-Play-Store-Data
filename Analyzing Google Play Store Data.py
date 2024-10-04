# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

# Download stopwords for sentiment analysis
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv('E:/oasis internship/Analyzing Google Play Store Data/datasets/apps.csv')

# Preview the dataset
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (optional step)
df.dropna(inplace=True)

# Data Cleaning: Fixing data types

# Clean 'Size' column
df['Size'] = df['Size'].apply(lambda x: x.replace('M', 'e6').replace('k', 'e3') if isinstance(x, str) else x)
df['Size'] = pd.to_numeric(df['Size'], errors='coerce')

# Clean 'Installs' column
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '').replace('+', '') if isinstance(x, str) else x)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

# Clean 'Price' column
df['Price'] = df['Price'].apply(lambda x: x.replace('$', '') if isinstance(x, str) else x)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop duplicates
df.drop_duplicates(inplace=True)

# Visualization: App Distribution Across Categories
plt.figure(figsize=(12, 8))
sns.countplot(y='Category', data=df, order=df['Category'].value_counts().index)
plt.title("App Distribution Across Categories")
plt.show()

# Top Categories
top_categories = df['Category'].value_counts().head(10)
print("Top 10 Categories by Number of Apps:")
print(top_categories)

# Metrics Analysis: App Ratings Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Rating'].dropna(), kde=False, bins=30)
plt.title('Distribution of App Ratings')
plt.show()

# App Size vs. Rating Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Size', y='Rating', data=df)
plt.title('App Size vs. Rating')
plt.show()

# Installs vs. Price Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='Installs', data=df)
plt.title('Price vs. Number of Installs')
plt.show()

# Sentiment Analysis on Reviews (if Reviews column exists)
# Example data processing assuming 'Reviews' column exists
if 'Reviews' in df.columns:
    df['Reviews'] = df['Reviews'].astype(str).apply(lambda x: x.lower())
    df['Reviews_clean'] = df['Reviews'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

    # Generate a word cloud from reviews
    wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(' '.join(df['Reviews_clean']))

    # Plot Word Cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of App Reviews')
    plt.show()

    # Perform sentiment analysis on reviews
    df['sentiment'] = df['Reviews'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Plot sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment'], bins=30, kde=True)
    plt.title('Sentiment Polarity Distribution in Reviews')
    plt.show()

# Interactive Visualizations with Plotly

# 1. Interactive Bar Chart: App Installs Across Categories
fig = px.bar(df, x='Category', y='Installs', title='Category vs. Installs', color='Category', log_y=True)
fig.show()

# 2. Interactive Scatter Plot: App Size vs Rating by Category
fig = px.scatter(df, x='Size', y='Rating', title='App Size vs Rating', color='Category', size='Installs')
fig.show()

# Optional: Save the cleaned dataset for further use
df.to_csv('E:/oasis internship/Analyzing Google Play Store Data/datasets/user_reviews.csv', index=False)
