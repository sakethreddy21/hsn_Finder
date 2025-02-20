import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer  # Hugging Face Transformers
from dotenv import load_dotenv
import os

# Initialize Pinecone
load_dotenv()

# Load the Excel file
file_path = "./HSN_Muthu_Db.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
index_name = "hsn-vector-db-deepseek"

# Delete the existing index if it exists
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

# Create a new index with the correct dimension (384 for all-MiniLM-L6-v2)
pc.create_index(
    name=index_name,
    dimension=384,  # Match the dimension of the Hugging Face model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region as needed
)

# Connect to the new Pinecone index
index = pc.Index(index_name)

# Load Hugging Face embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and effective model

# Function to generate text embeddings in batches using Hugging Face
def get_embeddings(texts, batch_size=50):
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        
        # Generate embeddings using Hugging Face
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)

    return embeddings

# Prepare and insert data into Pinecone in batches
batch_size = 100  # Adjust based on your dataset size
descriptions = df["Description"].astype(str).tolist()
hsn_codes = df["HSN CODES"].astype(str).tolist()

# Generate embeddings in batches
print("Generating embeddings in batches...")
embeddings = get_embeddings(descriptions, batch_size=50)

# Prepare vectors for Pinecone
vectors = [
    (hsn_codes[i], embeddings[i], {"hsn_code": hsn_codes[i], "description": descriptions[i]})
    for i in range(len(hsn_codes))
]

# Upsert data into Pinecone in batches
print("Uploading vectors to Pinecone...")
for i in range(0, len(vectors), batch_size):
    index.upsert(vectors[i : i + batch_size])
    print(f"Uploaded batch {i // batch_size + 1}")

print("HSN Codes and Descriptions uploaded successfully!")