# ingestion/build_index.py

from dotenv import load_dotenv
load_dotenv()

from ingestion.loaders import load_all_documents
from ingestion.chunker import split_documents
from ingestion.vectorstore import build_vectorstore

if __name__ == "__main__":
    print("Step 1: Loading documents...")
    docs = load_all_documents()

    print("Step 2: Splitting into chunks...")
    chunks = split_documents(docs)

    print(f"Step 3: Embedding {len(chunks)} chunks and building vector store...")
    build_vectorstore(chunks)

    print("Done. Vector store persisted to ./chroma_db")