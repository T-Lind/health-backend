import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from model import SuicideRiskClassifier, llm_predict

# TODO: when running move to root directory b/c of the relative path

def train_and_evaluate_ensemble(data_path, save_model=False, model_path='ensemble_model.pkl'):
    df = pd.read_csv(data_path)

    if 'Post' not in df.columns or 'Label' not in df.columns:
        raise ValueError("The dataset must contain 'Post' and 'Label' columns")

    label_encoder = SuicideRiskClassifier().le
    df['Encoded_Label'] = label_encoder.transform(df['Label'])

    # Split the data using the same random state to prevent evaluating on trained data!
    _, val_texts, _, val_labels = train_test_split(df['Post'], df['Encoded_Label'], test_size=0.2)

    val_df = pd.DataFrame({'Post': val_texts, 'Label': val_labels})

    classifier = SuicideRiskClassifier()

    val_df['BERT_Prediction'] = val_df['Post'].apply(classifier.bert_predict)
    val_df['LLM_Prediction'] = val_df['Post'].apply(llm_predict)
    val_df['Sentiment_Prediction'] = val_df['Post'].apply(classifier.sentiment_predict)


    val_df['BERT_Prediction'] = label_encoder.transform(val_df['BERT_Prediction'])
    val_df['LLM_Prediction'] = label_encoder.transform(val_df['LLM_Prediction'])
    val_df['Sentiment_Prediction'] = label_encoder.transform(val_df['Sentiment_Prediction'])

    X_val = val_df[['BERT_Prediction', 'LLM_Prediction', 'Sentiment_Prediction']]
    y_val = val_df['Label']

    ensemble_model = RandomForestClassifier(random_state=42)
    ensemble_model.fit(X_val, y_val)

    if save_model:
        joblib.dump(ensemble_model, model_path)
        print(f"Ensemble model saved to {model_path}")

    y_pred = ensemble_model.predict(X_val)
    print(classification_report(y_val, y_pred))

    conf_matrix = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Example usage
train_and_evaluate_ensemble('../data/500_Reddit_users_posts_labels.csv', save_model=True)