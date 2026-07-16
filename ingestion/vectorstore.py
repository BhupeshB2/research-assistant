# ingestion/vectorstore.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# Share the same folder path for saving and loading
PERSIST_DIRECTORY = "./chroma_db"

def get_embedding_model():
    """Helper to ensure we use the exact same embedding model for building and loading."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vectorstore(chunks):
    """Creates a brand new vector store from document chunks and saves it."""
    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore


def load_vectorstore():
    """Loads the already built vector store from disk."""
    embeddings = get_embedding_model()

    # This reads the existing database sitting in './chroma_db'
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vectorstore