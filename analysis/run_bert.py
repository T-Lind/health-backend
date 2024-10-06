import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# I've found that things often go wrong esp. when data is not loaded correctly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Original dataframe shape: {df.shape}")
    df['Post'] = df['Post'].apply(lambda x: str(x).replace('[', '').replace(']', '').replace("'", ''))
    logging.info(f"Processed dataframe shape: {df.shape}")
    return df


class SuicidePostDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        logging.info(f"Dataset initialized with {len(texts)} samples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        try:
            text = str(self.texts.iloc[item])
            label = self.labels.iloc[item]

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            logging.error(f"Error in __getitem__ for item {item}: {str(e)}")
            raise


def train_model(model, train_dataloader, val_dataloader, epochs, device):
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            except Exception as e:
                logging.error(f"Error in training loop: {str(e)}")
                continue

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                except Exception as e:
                    logging.error(f"Error in validation loop: {str(e)}")
                    continue

        logging.info(f'Epoch: {epoch + 1}, Val Loss: {val_loss / len(val_dataloader)}')


def predict(model, text, tokenizer, device, le):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

    predicted_label = le.inverse_transform(preds.cpu().numpy())[0]
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    max_prob = torch.max(probabilities).item()

    if max_prob < 0.5:  # TODO: adjust this threshold as needed.
        return "Inconclusive"
    else:
        return predicted_label


def main():
    df = load_data('../data/500_Reddit_users_posts_labels.csv')

    le = LabelEncoder()
    df['Encoded_Label'] = le.fit_transform(df['Label'])

    train_texts, val_texts, train_labels, val_labels = train_test_split(df['Post'], df['Encoded_Label'], test_size=0.2,
                                                                        random_state=42)

    logging.info(f"Train set size: {len(train_texts)}, Validation set size: {len(val_texts)}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(le.classes_))

    max_len = 512
    batch_size = 16

    train_dataset = SuicidePostDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_dataset = SuicidePostDataset(val_texts, val_labels, tokenizer, max_len=max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model.to(device)
    train_model(model, train_dataloader, val_dataloader, epochs=4, device=device)

    # Save model
    model.save_pretrained('slb-0002')
    tokenizer.save_pretrained('slb-0002')
    logging.info("Model and tokenizer saved successfully")


if __name__ == "__main__":
    main()
