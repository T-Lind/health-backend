import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('../data/500_Reddit_users_posts_labels.csv')

data['Post'] = data['Post'].str.join(' ')


le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

X_train, X_val, y_train, y_val = train_test_split(data['Post'], data['Label'], test_size=0.2, random_state=42)

empty_docs = X_train[X_train.str.strip() == '']
if not empty_docs.empty:
    print("Warning: There are empty documents in the training data.")
    X_train = X_train[X_train.str.strip() != '']

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
try:
    X_train_tfidf = tfidf.fit_transform(X_train)
except ValueError as e:
    print(f"Error: {e}")
    print("Possible reasons: documents only contain stop words or are empty.")
    problematic_docs = [doc for doc in X_train if len(tfidf.build_tokenizer()(doc)) == 0]
    # print("Problematic documents:", problematic_docs)
    # Remove problematic documents
    X_train = [doc for doc in X_train if len(tfidf.build_tokenizer()(doc)) > 0]
    X_train_tfidf = tfidf.fit_transform(X_train)

X_val_tfidf = tfidf.transform(X_val)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)
log_reg.fit(X_train_tfidf, y_train)
y_pred_log_reg = log_reg.predict(X_val_tfidf)

rf = RandomForestClassifier(class_weight=class_weights_dict, n_estimators=100, random_state=42)
rf.fit(X_train_tfidf, y_train)
y_pred_rf = rf.predict(X_val_tfidf)

svm = SVC(class_weight='balanced', kernel='linear')
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_val_tfidf)

print("Logistic Regression Classification Report:")
print(classification_report(y_val, y_pred_log_reg, target_names=le.classes_))

print("Random Forest Classification Report:")
print(classification_report(y_val, y_pred_rf, target_names=le.classes_))

print("SVM Classification Report:")
print(classification_report(y_val, y_pred_svm, target_names=le.classes_))

print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_val, y_pred_rf))