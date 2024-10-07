import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import LongformerTokenizer, LongformerForSequenceClassification, BigBirdTokenizer, BigBirdForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

df = pd.read_csv('../data/500_Reddit_users_posts_labels.csv')

if 'sentiment' not in df.columns:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['Post'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

df['post_length'] = df['Post'].apply(len)

X = df['Post']  # This will now contain the raw text
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

MAX_LEN = 4096  # Longformer/BigBird can handle up to 4096 tokens

model_choice = 'longformer'  # change this to 'bigbird' for BigBird

if model_choice == 'longformer':
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=5)
elif model_choice == 'bigbird':
    tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', num_labels=5)

def tokenize_function(examples):
    return tokenizer(examples['Post'], padding='max_length', truncation=True, max_length=MAX_LEN)

train_data = pd.DataFrame({'Post': X_train, 'Label': y_train})
test_data = pd.DataFrame({'Post': X_test, 'Label': y_test})

train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label'])

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

eval_results = trainer.evaluate()
print(eval_results)
