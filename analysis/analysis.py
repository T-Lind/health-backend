#%%
import pandas as pd

# Load the CSV data
data = pd.read_csv('../data/500_Reddit_users_posts_labels.csv')
#%%
from sklearn.feature_extraction.text import TfidfVectorizer

# Concatenate post lists into a single string for each user
data['Post'] = data['Post'].apply(lambda x: ' '.join(eval(x)))

# Check if the 'Post' column is empty after preprocessing
if data['Post'].str.strip().eq('').any():
    raise ValueError("Some posts are empty after preprocessing.")

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the posts
try:
    tfidf_features = tfidf.fit_transform(data['Post'])
except ValueError as e:
    print(f"Error: {e}")
    print("Check the preprocessing steps and ensure the documents contain valid words.")
#%%
from openai import OpenAI
import openai
from dotenv import load_dotenv
import os

print("Loaded env file: ", load_dotenv("../.env", override=True))

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Function to get embeddings for each post
def get_embeddings(post):
    response = client.embeddings.create(
        input=post,
        model="text-embedding-3-large",
    )
    return response.data[0].embedding

# Apply the embedding function to each concatenated post
data['embeddings'] = data['Post'].apply(lambda x: get_embeddings(x))

# Convert the embeddings to a matrix form (numpy array)
import numpy as np
embeddings_matrix = np.array(data['embeddings'].tolist())
#%%
print("Embeddings matrix shape:", embeddings_matrix.shape)
#%%
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Ensure all labels are strings
data['Label'] = data['Label'].astype(str)
print(np.unique(data['Label']))

# drop rows with missing labels
data = data.dropna(subset=['Label'])

# Compute class weights for the imbalance
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(data['Label']), 
    y=data['Label']
)

# Convert to dictionary form (useful for models like SVM or neural nets)
class_weights_dict = {label: weight for label, weight in zip(np.unique(data['Label']), class_weights)}
print(class_weights_dict)
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# Encode labels to numeric form
le = LabelEncoder()
y_encoded = le.fit_transform(data['Label'])

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(embeddings_matrix, y_encoded, test_size=0.2, random_state=42)


y_train = np.array(y_train)
y_val = np.array(y_val)

# Define the input shape based on the shape of the embeddings (dimensionality of embedding)
input_shape = X_train.shape[1:]

# Define the model
model = Sequential([
    Input(shape=input_shape),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')  # Assuming 5 classes in the output
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#%%
# ensure that X_train and y_train are proper to go into the model
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

print(X_train.dtype, y_train.dtype)
print(X_val.dtype, y_val.dtype)

print(np.unique(y_train), np.unique(y_val))
#%%
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, class_weight=class_weights_dict)
#%%
# Save the model
model.save('../models/nn_model.h5')