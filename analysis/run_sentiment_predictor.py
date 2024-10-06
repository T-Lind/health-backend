import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load the data
df = pd.read_csv('../data/500_Reddit_users_posts_labels.csv')

# Ensure the sentiment column is present
if 'sentiment' not in df.columns:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['Post'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Add post length
df['post_length'] = df['Post'].apply(len)

# Prepare the data
X = df[['sentiment', 'post_length']]
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, '../models/sentiment_predictor.pkl')