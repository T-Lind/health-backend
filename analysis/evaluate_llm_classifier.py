import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from model import llm_predict

# TODO: should be in root directory b/c model will try to import system prompt incorrectly.
#  simply here for repo cleanliness

def evaluate_llm_classifier(data_path, fraction=1.0):
    # Load the data
    df = pd.read_csv(data_path)

    # Ensure the 'Post' and 'Label' columns are present
    if 'Post' not in df.columns or 'Label' not in df.columns:
        raise ValueError("The dataset must contain 'Post' and 'Label' columns")

    # Sample the dataset
    df = df.sample(frac=fraction, random_state=42)

    # Predict labels using llm_predict
    df['Predicted_Label'] = df['Post'].apply(llm_predict)

    # Generate the classification report
    print(classification_report(df['Label'], df['Predicted_Label']))

    # Plot the confusion matrix
    conf_matrix = confusion_matrix(df['Label'], df['Predicted_Label'])
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=df['Label'].unique(), yticklabels=df['Label'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print up to 5 cases where the model gets it wrong
    wrong_predictions = df[df['Label'] != df['Predicted_Label']]
    print("Cases where the model gets it wrong:")
    for index, row in wrong_predictions.head(5).iterrows():
        print(f"Post: {row['Post']}\nActual Label: {row['Label']}\nPredicted Label: {row['Predicted_Label']}\n")

# Example usage
evaluate_llm_classifier('../data/500_Reddit_users_posts_labels.csv', fraction=0.2)