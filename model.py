import torch
from transformers import BertTokenizer, BertForSequenceClassification
import random

class RandomClassifier:
    def __init__(self):
        pass

    def predict(self, text):
        """Simply give a random output."""
        return random.choice(['Attempt', 'Behavior', 'Ideation', 'Indicator', 'Supportive'])

class SuicideRiskClassifier:
    def __init__(self, model_path='fine_tuned_bert', max_length=128, device=None):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(model_path)

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.max_length = max_length

        self.label_map = {
            0: 'Attempt',
            1: 'Behavior',
            2: 'Ideation',
            3: 'Indicator',
            4: 'Supportive'
        }

    def preprocess(self, text):
        """Tokenize the input text for BERT."""
        encodings = self.tokenizer(text, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        encodings = {key: val.to(self.device) for key, val in encodings.items()}
        return encodings

    def predict(self, text):
        """Given a single piece of text, predict the suicide risk label."""
        self.model.eval()
        encodings = self.preprocess(text)

        with torch.no_grad():
            outputs = self.model(**encodings)

        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

        return self.label_map[predicted_class_id]

    def predict_batch(self, texts):
        """Given a list of texts, predict the suicide risk label for each."""
        self.model.eval()
        predictions = []

        for text in texts:
            predictions.append(self.predict(text))

        return predictions

    def evaluate(self, texts, true_labels):
        """Evaluate the model on a batch of texts with true labels for comparison."""
        predictions = self.predict_batch(texts)

        true_labels = [self.label_map[label] for label in true_labels]

        from sklearn.metrics import classification_report, confusion_matrix
        print("Classification Report:")
        print(classification_report(true_labels, predictions))

        print("Confusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
