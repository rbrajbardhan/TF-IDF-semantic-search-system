import streamlit as st
import pandas as pd
import re
import nltk
import joblib

from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Indian Parliament Semantic Search",
    layout="wide"
)

st.title("🇮🇳 Indian Parliament Q&A Semantic Search")
st.caption("TF-IDF + Cosine Similarity based semantic retrieval")

# ===============================
# NLTK Setup (Cached)
# ===============================
@st.cache_resource
def setup_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")

setup_nltk()
STOP_WORDS = set(stopwords.words("english"))

# ===============================
# Utility Functions
# ===============================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

def highlight_text(text: str, keywords: list) -> str:
    if not text or not keywords:
        return text

    highlighted = text
    for word in set(keywords):
        pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
        highlighted = pattern.sub(r"<mark>\1</mark>", highlighted)

    return highlighted

# ===============================
# Load Saved Model Artifacts
# ===============================
@st.cache_resource
def load_model():
    vectorizer = joblib.load("model/tfidf_vectorizer.joblib")
    tfidf_matrix = joblib.load("model/tfidf_matrix.joblib")
    df = joblib.load("model/processed_dataframe.joblib")
    return vectorizer, tfidf_matrix, df

vectorizer, tfidf_matrix, df = load_model()

# ===============================
# Sidebar Filters (AUTO-SYNCED)
# ===============================
st.sidebar.header("🔍 Filters")

top_k = st.sidebar.slider("Number of results", 1, 10, 5)

selected_ministry = st.sidebar.multiselect(
    "Ministry",
    sorted(df["ministry"].dropna().unique())
)

selected_years = st.sidebar.multiselect(
    "Year",
    sorted(df["year"].dropna().unique())
)

default_start = df["answer_date"].min()
default_end = df["answer_date"].max()

if selected_years:
    start_date = pd.Timestamp(f"{min(selected_years)}-01-01")
    end_date = pd.Timestamp(f"{max(selected_years)}-12-31")

    st.sidebar.date_input(
        "Date Range (auto-adjusted)",
        value=(start_date.date(), end_date.date()),
        disabled=True
    )
else:
    start_date, end_date = st.sidebar.date_input(
        "Date Range",
        value=(default_start.date(), default_end.date()),
        min_value=default_start.date(),
        max_value=default_end.date()
    )

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# ===============================
# Search Interface
# ===============================
query = st.text_input(
    "Enter your search query",
    placeholder="e.g. women safety, rural electrification, handloom rebate"
)

if query:
    cleaned_query = clean_text(query)
    query_tokens = cleaned_query.split()

    query_vec = vectorizer.transform([cleaned_query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    results = df.copy()
    results["score"] = scores
    results = results.sort_values("score", ascending=False)

    if selected_ministry:
        results = results[results["ministry"].isin(selected_ministry)]

    if selected_years:
        results = results[results["year"].isin(selected_years)]

    results = results[
        (results["answer_date"] >= start_date) &
        (results["answer_date"] <= end_date)
    ]

    results = results.head(top_k)

    st.subheader("📄 Search Results")

    if results.empty:
        st.warning("No results found for the selected query and filters.")
    else:
        for _, row in results.iterrows():
            with st.expander(row["question_title"]):
                st.markdown(f"**Ministry:** {row['ministry']}")
                st.markdown(f"**Date:** {row['answer_date'].date()}")
                st.markdown(f"**Question By:** {row['question_by']}")
                st.markdown("---")

                st.markdown("**Question:**", unsafe_allow_html=True)
                st.markdown(
                    highlight_text(row["question_description"], query_tokens),
                    unsafe_allow_html=True
                )

                st.markdown("**Answer:**", unsafe_allow_html=True)
                st.markdown(
                    highlight_text(row["answer"], query_tokens),
                    unsafe_allow_html=True
                )

                st.markdown(f"**Similarity Score:** `{row['score']:.4f}`")
else:
    st.info("Enter a query above to start searching.")
