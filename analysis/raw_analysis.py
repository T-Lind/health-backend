import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

# nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv('../data/500_Reddit_users_posts_labels.csv')

# Show the class distribution
plt.figure(figsize=(10, 6))
sns.countplot(data['Label'])
plt.title('Class Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

data['Post_Length'] = data['Post'].apply(lambda x: len(word_tokenize(x)))

plt.figure(figsize=(10, 6))
sns.histplot(data['Post_Length'], bins=30, kde=True)
plt.title('Post Length Distribution')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# Also print the results of the plots
print(data['Label'].value_counts())
print(data['Post_Length'].describe())


def get_common_words(posts, num_words=10):
    all_words = ' '.join(posts).lower()
    tokens = word_tokenize(all_words)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    word_freq = Counter(filtered_tokens)
    common_words = word_freq.most_common(num_words)
    return common_words

labels = data['Label'].unique()
common_words_by_label = {}

for label in labels:
    posts = data[data['Label'] == label]['Post']
    common_words_by_label[label] = get_common_words(posts)

for label, common_words in common_words_by_label.items():
    print(f"Common words in {label} posts:")
    for word, freq in common_words:
        print(f"{word}: {freq}")
    print("\n")