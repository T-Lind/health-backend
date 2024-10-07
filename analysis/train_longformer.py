import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import LongformerTokenizer, LongformerForSequenceClassification, BigBirdTokenizer, BigBirdForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from datasets import Dataset

# Load the data
df = pd.read_csv('../data/500_Reddit_users_posts_labels.csv')

# Ensure sentiment column is present (VADER sentiment analysis)
if 'sentiment' not in df.columns:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['Post'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Add post length
df['post_length'] = df['Post'].apply(len)

# Prepare the data for classification
X = df['Post']  # This will now contain the raw text
y = df['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization and Model Preparation
MAX_LEN = 4096  # Longformer/BigBird can handle up to 4096 tokens

# Choose the model: Longformer or BigBird
model_choice = 'longformer'  # change this to 'bigbird' for BigBird

if model_choice == 'longformer':
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=5)
elif model_choice == 'bigbird':
    tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', num_labels=5)

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['Post'], padding='max_length', truncation=True, max_length=MAX_LEN)

# Convert the data into Hugging Face Dataset format
train_data = pd.DataFrame({'Post': X_train, 'Label': y_train})
test_data = pd.DataFrame({'Post': X_test, 'Label': y_test})

# Create Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set the dataset format to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label'])

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",      # Evaluate at the end of each epoch
    save_strategy="epoch",            # Save model at the end of each epoch to match evaluation
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,      # Load the best model at the end of training
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)
