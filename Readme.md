# Research Assistant (LangChain)

A retrieval-augmented research assistant that answers questions grounded in your own documents (PDF, DOCX, TXT, MD) — with cited sources, conversation memory, and full request tracing via LangSmith.

## What it does

Upload internal documents (reports, policies, notes) into `data/`, run the ingestion pipeline once, then ask questions in a CLI chat loop. The assistant retrieves relevant chunks from those documents, answers only from that retrieved context, and returns the answer alongside the source filenames it used. If the documents don't contain enough information, it says so instead of guessing.

## Architecture

```
data/ (your files)
   → ingestion/loaders.py     (file → LangChain Document objects)
   → ingestion/chunker.py     (Document → smaller chunks)
   → ingestion/vectorstore.py (chunks → embeddings → Chroma vector store)
   → app/chain.py             (retriever → prompt → structured LLM output)
   → main.py                  (CLI loop with conversation memory)
```

Ingestion (loading, chunking, embedding) runs once via `ingestion/build_index.py`, separate from answering questions — the vector store is only rebuilt when documents change.

## Project structure

```
research-assistant/
├── data/                  # drop your PDFs/docx/txt/md files here
├── ingestion/
│   ├── loaders.py         # routes each file type to the right loader
│   ├── chunker.py         # splits documents into retrievable chunks
│   ├── vectorstore.py     # builds/loads the Chroma vector store
│   └── build_index.py     # one-time ingestion script
├── app/
│   └── chain.py           # retrieval chain, prompt, structured output, memory
├── main.py                # CLI entry point
├── requirements.txt
├── .env                   # API keys (not committed)
└── .gitignore
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here

# Optional — enables tracing in LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=research-assistant
```

## Usage

1. Drop your source documents into `data/` (PDF, DOCX, TXT, or MD)
2. Build the vector store (run this once, and again any time documents change):
   ```bash
   python -m ingestion.build_index
   ```
3. Start the assistant:
   ```bash
   python main.py
   ```
4. Ask questions. Type `exit` to quit.

## Configuration knobs

| Setting | Location | Default | Effect |
|---|---|---|---|
| Chunk size / overlap | `ingestion/chunker.py` | 800 / 150 chars | Larger = more context per chunk but less precise retrieval |
| Retrieved chunks (`k`) | `app/chain.py` | 4 | Higher = more recall, more noise, more latency/cost |
| Model | `app/chain.py` | `gpt-4o-mini` | Swap for a larger model if answer quality needs improvement |

## Monitoring

With the `LANGCHAIN_*` variables set in `.env`, every question is automatically traced in LangSmith — showing retrieved chunks, the exact prompt sent, the raw model response, and per-step latency. No code changes needed to enable this.

## Roadmap / possible extensions

- Add a web-search tool alongside document retrieval and upgrade the chain into a full tool-calling agent
- Add persistent (cross-session) memory instead of in-memory chat history
- Add a re-ranking step on top of initial retrieval

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: research assistant with LangChain RAG pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/research-assistant.git
git push -u origin main
```

`.env` and `chroma_db/` are already excluded via `.gitignore` — double check before pushing that no API keys are committed.