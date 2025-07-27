import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import preprocess_pipeline

# Optional: set data paths
RAW_DATA_PATH = os.path.join('..', 'data', 'raw', 'news_raw.csv')
CLEAN_DATA_PATH = os.path.join('..', 'data', 'processed', 'news_clean.csv')
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

tqdm.pandas()

def main():
    # 1. Load data
    print('Loading data...')
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.dropna(subset=['text', 'label']).drop_duplicates(subset=['text']).reset_index(drop=True)

    # Optionally use title + text
    # df['text'] = df['title'].astype(str) + " " + df['text'].astype(str)

    # 2. Preprocess text
    print('Preprocessing...')
    df['text_clean'] = df['text'].progress_apply(preprocess_pipeline)

    # Save cleaned data
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    # 3. Train/test split
    X = df['text_clean']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Vectorization
    print('Vectorizing...')
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Train model (XGBoost)
    
    print('Training model...')
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    model.fit(X_train_vec, y_train)

    # 6. Save model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print('Saved model and vectorizer.')

    # 7. Evaluate
    print('Evaluating...')
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    main()
