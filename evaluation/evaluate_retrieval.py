import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("parliament_merged.csv")

df["document"] = (
    df["question_title"].fillna("") + " " +
    df["question_description"].fillna("") + " " +
    df["answer"].fillna("")
).str.lower()

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(df["document"])

queries = [
    "women safety",
    "rural electrification",
    "education policy",
    "agricultural subsidy"
]

print("Qualitative Evaluation Results\n")

for query in queries:
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[-3:][::-1]

    print(f"\nQuery: {query}")
    for idx in top_idx:
        print("-", df.iloc[idx]["question_title"])
