import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load files
embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()


def retrieve_top_k(query_vector, k=5):
    similarities = cosine_similarity(
        query_vector.reshape(1, -1),
        embeddings
    )[0]

    top_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_indices]


def get_query_vector(query):
    query_words = query.lower().split()

    matched_indices = [
        i for i, doc in enumerate(documents)
        if any(word in doc.lower() for word in query_words)
    ]

    if matched_indices:
        return np.mean(embeddings[matched_indices], axis=0)

    return np.zeros(embeddings.shape[1])


def get_top_sentences(doc_text, query, top_n=3):
    query_words = query.lower().split()
    sentences = doc_text.replace("\n", " ").split(".")

    scored = []

    for sentence in sentences:
        score = sum(word in sentence.lower() for word in query_words)
        scored.append((sentence.strip(), score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# UI
st.title("Information Retrieval System")

query = st.text_input("Enter your query:")

if st.button("Search") and query:

    query_vector = get_query_vector(query)
    results = retrieve_top_k(query_vector)

    st.write("### Top Relevant Documents:")

    for doc_text, score in results:

        st.write(f"## Document (Score: {score:.4f})")

        top_sentences = get_top_sentences(doc_text, query)

        for sentence, sent_score in top_sentences:
            if sent_score > 0:
                st.write(f"- {sentence}")

        st.write("---")