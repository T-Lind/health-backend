import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from transformers import BertTokenizerFast
from tqdm import tqdm

data = pd.read_csv('../data/500_Reddit_users_posts_labels.csv')

# join the list of posts into a single string
data['Post'] = data['Post'].str.join(' ')

# Drop rows with missing labels
data = data.dropna(subset=['Label'])
print("Unique labels:", data['Label'].unique())
print("Number of entries:", len(data))

le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

X_train, X_val, y_train, y_val = train_test_split(data['Post'], data['Label'], test_size=0.2, random_state=42)

y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

MAX_LENGTH = 512

train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=MAX_LENGTH)
val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=MAX_LENGTH)

print("Length of train_encodings['input_ids']:", len(train_encodings['input_ids']))
print("Length of y_train:", len(y_train))


class SuicideRiskDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = np.array(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert labels to LongTensor
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SuicideRiskDataset(train_encodings, y_train)
val_dataset = SuicideRiskDataset(val_encodings, y_val)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device: ", device)
model.to(device)

# Define the DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 3  # 3 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# convert y_train to numpy array
y_train = np.array(y_train)

assert len(train_encodings['input_ids']) == len(y_train), "Mismatch between tokenized data and labels!"
assert isinstance(y_train, np.ndarray), "Labels must be a numpy array!"



# Training loop
model.train()
for epoch in range(3):  # Train for 3 epochs
    print(f"Epoch {epoch+1}/{3}")
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_bert')

# Evaluation
model.eval()
predictions, true_labels = [], []
for batch in val_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions.append(logits.argmax(dim=-1).cpu().numpy())
    true_labels.append(batch['labels'].cpu().numpy())

predictions = np.concatenate(predictions)
true_labels = np.concatenate(true_labels)

print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(true_labels, predictions))
