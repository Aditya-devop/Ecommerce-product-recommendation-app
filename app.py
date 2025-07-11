import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- CSS Styling --------------------
st.markdown("""
    <style>
        body {
            background-color: #f1f3f6;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            color: #2c3e50;
            font-size: 3em;
            font-weight: 700;
            text-align: center;
            margin-bottom: 20px;
        }
        .recommendation {
            display: flex;
            flex-direction: row;
            background: linear-gradient(to right, #ffffff, #f9f9f9);
            border: 1px solid #ddd;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.07);
        }
        .recommendation img {
            width: 150px;
            height: 150px;
            object-fit: contain;
            border-radius: 10px;
            margin-right: 20px;
            border: 1px solid #ccc;
            background: #fff;
        }
        .info {
            color: #2d3436;
            font-size: 1.1em;
        }
        .similarity {
            color: #0984e3;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- App Title --------------------
st.markdown("<div class='title'>🛍️ E-Commerce Product Recommender</div>", unsafe_allow_html=True)

# -------------------- Load Precomputed Dataset --------------------
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

# -------------------- Recommendation Function --------------------
def recommend_products(query, top_k=5):
    query = query.lower()
    query_embedding = model.encode(query)
    df['similarity'] = df['embeddings'].apply(lambda x: cosine_similarity([query_embedding], [x]).flatten()[0])
    return df.sort_values(by='similarity', ascending=False).head(top_k)

# -------------------- Search Input --------------------
query = st.text_input("🔍 Search for products (e.g. '8GB RAM smartphone')", '')

if query:
    results = recommend_products(query)

    if results.empty:
        st.warning("No products found.")
    else:
        st.markdown("### 🔎 Top Recommendations:")
        for _, row in results.iterrows():
            img_list = eval(row['imgs']) if isinstance(row['imgs'], str) else row['imgs']
            image_url = img_list[0] if isinstance(img_list, list) and len(img_list) > 0 else ""

            st.markdown(f"""
                <div class='recommendation'>
                    <img src="{image_url}" alt="Product Image">
                    <div class='info'>
                        <b>Title:</b> {row['title']}<br>
                        <b>Brand:</b> {row['brand']}<br>
                        <b>Category:</b> {row['category']}<br>
                        <b>Similarity Score:</b> <span class='similarity'>{row['similarity']:.2f}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
