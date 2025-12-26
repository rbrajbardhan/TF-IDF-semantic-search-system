import pandas as pd
import matplotlib.pyplot as plt
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

queries = ["women safety", "rural electrification", "education policy"]

plt.figure()
for query in queries:
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_scores = sorted(scores, reverse=True)[:10]
    plt.plot(range(1, 11), top_scores, label=query)

plt.xlabel("Rank")
plt.ylabel("Cosine Similarity Score")
plt.title("Top-10 Similarity Score Comparison Across Queries")
plt.legend()
plt.show()
