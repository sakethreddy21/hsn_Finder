import streamlit as st
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
index_name = "hsn-vector-db-deepseek"

# Connect to Pinecone index
index = pc.Index(index_name)

# Load the same embedding model used for indexing
model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("📌 HSN Code Finder Chatbot")

st.markdown("Enter a product description to find the closest matching HSN code.")

# Chat input
user_query = st.text_input("🔍 Enter Product Description:", "")

if user_query:
    # Get query embedding
    query_embedding = model.encode(user_query).tolist()
    
    # Search Pinecone
    search_results = index.query(vector=query_embedding, top_k=1, include_metadata=True)

    # Display results
    if search_results["matches"]:
        match = search_results["matches"][0]
        hsn_code = match["id"]
        description = match["metadata"]["description"]

        st.success(f"✅ **HSN Code:** {hsn_code}")
        st.write(f"📝 **Description:** {description}")
    else:
        st.error("No matching HSN Code found. Try a different description!")

st.markdown("---")
st.markdown("🚀 **Powered by Pinecone & Hugging Face Embeddings**")
