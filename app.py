import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed document embeddings
embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()


def retrieve_top_k(query_embedding, embeddings, k=10):
    """Retrieve top-k most similar documents using cosine similarity."""
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]


# Keep your simple query embedding
def get_query_embedding(query):
    return np.random.rand(embeddings.shape[1])


# ➕ NEW: Get 1 best sentence (keyword-based only)
def get_best_sentence(doc_text, query):
    query_words = query.lower().split()

    sentences = doc_text.replace("\n", " ").split(".")

    best_sentence = ""
    best_score = -1

    for sentence in sentences:
        score = sum(word in sentence.lower() for word in query_words)

        if score > best_score:
            best_score = score
            best_sentence = sentence.strip()

    return best_sentence


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Information Retrieval using Document Embeddings")

query = st.text_input("Enter your query:")

if st.button("Search") and query:

    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings)

    st.write("### Top 10 Relevant Documents:")

    for doc, score in results:

        st.write(f"## Document (Score: {score:.4f})")

        # ➕ Show 1 best sentence
        best_sentence = get_best_sentence(doc, query)

        if best_sentence:
            st.write(f"➡ {best_sentence}")

        st.write("---")
