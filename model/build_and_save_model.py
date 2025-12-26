import pandas as pd
import re
import nltk
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK setup
nltk.download("punkt")
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

# Load dataset
df = pd.read_csv("../parliament_merged.csv")

df["answer_date"] = pd.to_datetime(
    df["answer_date"],
    format="%d.%m.%Y",
    errors="coerce"
)
df["year"] = df["answer_date"].dt.year

df["document"] = (
    df["question_title"].fillna("") + " " +
    df["question_description"].fillna("") + " " +
    df["answer"].fillna("")
)

df["clean_document"] = df["document"].apply(clean_text)

# Build TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(df["clean_document"])

# Save everything
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(tfidf_matrix, "tfidf_matrix.joblib")
joblib.dump(df, "processed_dataframe.joblib")

print("Model artifacts saved successfully.")
