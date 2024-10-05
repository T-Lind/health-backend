import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
import openai
from dotenv import load_dotenv

load_dotenv("../.env", override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt')
nltk.download('stopwords')

client = openai.OpenAI()

def get_embeddings(post):
    response = client.embeddings.create(
        input=post,
        model="text-embedding-3-large",
    )
    return response.data[0].embedding

def load_data(file_path):
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    df['Post'] = df['Post'].apply(lambda x: str(x).replace('[', '').replace(']', '').replace("'", ''))
    logging.info(f"Loaded dataframe shape: {df.shape}")
    return df

def get_common_words(text, n=10):
    stop_words = set(stopwords.words('english')).union({"im", "dont", "ive", "youre"})
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return Counter(words).most_common(n)

def print_sentiment_scores(df):
    logging.info("Sentiment Scores:")
    for index, row in df.iterrows():
        logging.info(f"Post: {row['Post']}\nSentiment Score: {row['sentiment']}\n")

def plot_sentiment_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment'], bins=30, kde=True)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.savefig('sentiment_distribution.png')
    plt.close()

def analyze_text(df):
    logging.info("Generating embeddings")
    # df['embeddings'] = df['Post'].apply(get_embeddings)
    #
    # logging.info("Performing PCA")
    # embeddings_matrix = np.vstack(df['embeddings'].values)
    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(embeddings_matrix)
    #
    # silhouette_avg = silhouette_score(embeddings_matrix, df['Label'])
    # logging.info(f"Silhouette Score: {silhouette_avg}")
    #
    # plt.figure(figsize=(12, 8))
    # sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['Label'])
    # plt.title('PCA of Post Embeddings')
    # plt.savefig('pca_plot.png')
    # plt.close()

    df['post_length'] = df['Post'].apply(len)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Label', y='post_length', data=df)
    plt.title('Post Length by Label')
    plt.savefig('post_length_boxplot.png')
    plt.close()

    df['avg_word_size'] = df['Post'].apply(lambda x: np.mean([len(word) for word in x.split()]))
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Label', y='avg_word_size', data=df)
    plt.title('Average Word Size by Label')
    plt.savefig('avg_word_size_boxplot.png')
    plt.close()

    df['common_words'] = df['Post'].apply(get_common_words)
    all_common_words = Counter()
    for words in df['common_words']:
        all_common_words.update(dict(words))

    plt.figure(figsize=(12, 6))
    words, counts = zip(*all_common_words.most_common(20))
    sns.barplot(x=list(words), y=list(counts))
    plt.title('20 Most Common Words')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('common_words_barplot.png')
    plt.close()

    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['Post'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    print_sentiment_scores(df)
    plot_sentiment_distribution(df)

    df.to_csv('processed_data.csv', index=False)
    logging.info("Analysis complete. Results saved as CSV and PNG files.")
    # Class distribution plot
    plt.figure(figsize=(10, 6))
    sns.countplot(df['Label'])
    plt.title('Class Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig('class_distribution.png')
    plt.close()

    # Post length distribution plot
    df['Post_Length'] = df['Post'].apply(lambda x: len(word_tokenize(x)))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Post_Length'], bins=30, kde=True)
    plt.title('Post Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.savefig('post_length_distribution.png')
    plt.close()

def main():
    df = load_data('../data/500_Reddit_users_posts_labels.csv')
    analyze_text(df)

if __name__ == "__main__":
    main()