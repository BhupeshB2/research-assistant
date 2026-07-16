# ingestion/chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: list, chunk_size: int = 200, chunk_overlap: int = 350) -> list:
    """
    Splits documents into smaller chunks for embedding and retrieval.
    """
    
    # 1. Check if we received a list of lists and flatten it
    flat_documents = []
    for doc in documents:
        if isinstance(doc, list):
            flat_documents.extend(doc)
        else:
            flat_documents.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    # 2. Pass the flattened list to the splitter
    chunks = splitter.split_documents(flat_documents)
    return chunks