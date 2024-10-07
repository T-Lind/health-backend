import torch
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

load_dotenv(".env", override=True)


class StringOutput(BaseModel):
    value: Literal["Attempt", "Behavior", "Ideation", "Indicator", "Supportive"] = Field(
        description="The label of type of suicidal behavior or action predicted by the model."
    )


llm = ChatOpenAI(model="gpt-4o-mini")

with open("./prompts/structured_llm_system_prompt.txt", "r", encoding="utf-8") as f:
    structured_llm_system_prompt = f.read()
messages = [SystemMessage(content=structured_llm_system_prompt), ]

structured_llm = llm.with_structured_output(StringOutput, method="json_schema")


def llm_predict(text, max_length=128):
    """ This is for demo purposes. Of course this isn't a 'ML Model' but it can be cheaper to use, actually"""
    limited_text = text[:max_length * 4]  # roughly 512 tokens
    response = structured_llm.invoke(messages + [HumanMessage(content="Label this text: " + limited_text)])
    return response.value


class SuicideRiskClassifier:
    def __init__(self, model_path='models/slb-0001', max_length=128, device=None, uncertain_threshold=0.5):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(model_path)

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.max_length = max_length
        self.le = LabelEncoder()

        self.le.fit(['Attempt', 'Behavior', 'Ideation', 'Indicator', 'Supportive'])

        self.uncertain_threshold = uncertain_threshold

        self.analyzer = SentimentIntensityAnalyzer()
        self.sentiment_predictor = joblib.load('models/sentiment_predictor.pkl')

    def preprocess(self, text):
        """Tokenize the input text for BERT."""
        encodings = self.tokenizer(text, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        encodings = {key: val.to(self.device) for key, val in encodings.items()}
        return encodings

    def bert_predict(self, text, max_length=512):
        """Given a single piece of text, predict the suicide risk label. 42% accuracy standalone. """
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

        predicted_label = self.le.inverse_transform(preds.cpu().numpy())[0]
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        max_prob = torch.max(probabilities).item()

        return predicted_label

    def sentiment_predict(self, text):
        """Predict the category based on sentiment score and post length. 27% accuracy standalone. """
        sentiment_score = self.analyzer.polarity_scores(text)['compound']
        post_length = len(text)
        predicted_label = self.sentiment_predictor.predict([[sentiment_score, post_length]])[0]
        return predicted_label

    def ensemble_predict(self, text):
        """Predict the class of the input text using the ensemble model and outputs from bert, llm, sentiment prediction."""
        # Generate predictions from each model
        bert_prediction = self.bert_predict(text)
        llm_prediction = llm_predict(text)
        sentiment_prediction = self.sentiment_predict(text)

        # Encode the predictions
        bert_encoded = self.le.transform([bert_prediction])[0]
        llm_encoded = self.le.transform([llm_prediction])[0]
        sentiment_encoded = self.le.transform([sentiment_prediction])[0]

        X = [[bert_encoded, llm_encoded, sentiment_encoded]]

        ensemble_model = joblib.load('models/ensemble_model.pkl')

        y_pred = ensemble_model.predict(X)
        predicted_label = self.le.inverse_transform(y_pred)[0]

        return predicted_label
