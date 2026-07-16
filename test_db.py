# test_db.py
from ingestion.vectorstore import load_vectorstore

# 1. Load the database
db = load_vectorstore()

# 2. Grab all stored documents
data = db.get()

print(f"Total chunks stored in DB: {len(data['documents'])}")

# 3. Print a few sample chunks and their sources
for i in range(min(5, len(data['documents']))):
    source = data['metadatas'][i].get('source_file', 'unknown')
    content = data['documents'][i][:150].replace('\n', ' ')
    print(f"\n[Chunk {i+1}] Source: {source}")
    print(f"Text Preview: {content}...")