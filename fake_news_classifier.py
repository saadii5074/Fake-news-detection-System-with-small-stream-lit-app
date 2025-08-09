import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# =======================
# 1. Load Dataset
# =======================
true_df = pd.read_csv(r"C:\Users\isaad\Downloads\True.csv\True.csv")

fake_df = pd.read_csv(r"C:\Users\isaad\Downloads\Fake.csv\Fake.csv")

# Add labels
true_df['label'] = 1  # Real
fake_df['label'] = 0  # Fake

# Combine datasets
df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)

# =======================
# 2. Text Preprocessing
# =======================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize & remove stopwords
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

df['text'] = df['title'] + " " + df['text']
df['text'] = df['text'].apply(clean_text)

# =======================
# 3. Train-Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =======================
# 4. Model Training
# =======================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# Save best model (Example: Logistic Regression)
import joblib
joblib.dump(models["Logistic Regression"], "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
