import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# -----------------------------
# Step 1: Load PDF files
# -----------------------------
def load_pdf_files(data_path):
    print("Loading PDF files...")
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(f"No PDF files found in {data_path}")
    print(f"Loaded {len(documents)} PDF pages.")
    return documents

# -----------------------------
# Step 2: Split documents into chunks
# -----------------------------
def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    return chunks

# -----------------------------
# Step 3: Initialize embedding model
# -----------------------------
def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print("Initializing HuggingFace embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Embedding model ready!")
    return embeddings

# -----------------------------
# Step 4: Create or load FAISS vectorstore
# -----------------------------
def create_or_load_faiss(chunks, embeddings, db_path):
    if os.path.exists(db_path):
        print(f"Loading existing FAISS vectorstore from {db_path}...")
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print("FAISS vectorstore loaded successfully.")
    else:
        print("Creating FAISS vectorstore from chunks...")
        db = FAISS.from_documents(chunks, embeddings)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db.save_local(db_path)
        print(f"FAISS vectorstore saved at {db_path}.")
    return db

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)
    embedding_model = get_embedding_model()
    db = create_or_load_faiss(text_chunks, embedding_model, DB_FAISS_PATH)
    print("All steps completed successfully!")
