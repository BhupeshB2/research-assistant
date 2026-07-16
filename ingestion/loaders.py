from pathlib import Path
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

#You might wonder Why a Dictionary? Well, this is because to 
# their respective loaders. This way, when we encounter a file with a specific extension, 
# we can easily find the appropriate loader to use for that file type. 
# This makes our code more organized and easier to maintain, 
# as we can simply add new file types and their loaders to this dictionary 
# without having to modify the rest of the code.
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_document(file_path:str) -> list:
    """
    Routes a single path to the correct Langchain Loader based on extension,
    and returns a list of Document objects ( text + metadata ) for that file.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in LOADER_MAP:
        raise ValueError(f"Unsupported file type: {extension}")

    loader_class = LOADER_MAP[extension]
    loader = loader_class(str(path))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_file"] = path.name

    return documents


def load_all_documents(data_dir : str = "data") -> list:
    """
    Loads every supported file in the data directory.
    """
    all_docs=[]
    data_path = Path(data_dir)

    for file_path in data_path.iterdir():
        if file_path.suffix.lower() in LOADER_MAP:
            print(f"Loading {file_path.name}")
            doc = load_document(str(file_path))
            all_docs.extend(doc)
        else:
            print(f"Skipping unsupported file type: {file_path.name}")
    return all_docs