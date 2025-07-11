import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import io

# -------------------- CSS Styling --------------------
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            color: #2c3e50;
            font-size: 3em;
            font-weight: 700;
            text-align: center;
            margin-bottom: 10px;
        }
        .recommendation {
            border: 1px solid #e0e0e0;
            border-radius: 15px;
            padding: 15px;
            background-color: white;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .recommendation img {
            max-width: 100%;
            border-radius: 10px;
        }
        .info {
            font-size: 1.1em;
        }
        .similarity {
            color: #16a085;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- App Title --------------------
st.markdown("<div class='title'>🛍️ E-Commerce Product Recommender</div>", unsafe_allow_html=True)

# -------------------- Load Precomputed Dataset (.pkl from same folder) --------------------
@st.cache_data(show_spinner=True)
def load_data():
    with open("product_embeddings.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()

# -------------------- Load Sentence-BERT Model --------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------- Product Recommendation Function --------------------
def recommend_products(query, top_k=5):
    query = query.lower()
    query_embedding = model.encode(query)

    df['similarity'] = df['embeddings'].apply(lambda x: cosine_similarity([query_embedding], [x]).flatten()[0])
    recommendations = df.sort_values(by='similarity', ascending=False).head(top_k)
    return recommendations

# -------------------- Search Box --------------------
query = st.text_input("🔍 Search for products (e.g. '8GB RAM smartphone')", '')

if query:
    results = recommend_products(query)

    if results.empty:
        st.warning("No products found.")
    else:
        st.markdown("### 🔎 Top Recommendations:")
        for _, row in results.iterrows():
            st.markdown(f"""
            <div class='recommendation'>
                <img src="{row['imgs']}" alt="Product Image">
                <div class='info'>
                    <b>Title:</b> {row['title']}<br>
                    <b>Brand:</b> {row['brand']}<br>
                    <b>Category:</b> {row['category']}<br>
                    <b>Similarity Score:</b> <span class='similarity'>{row['similarity']:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
