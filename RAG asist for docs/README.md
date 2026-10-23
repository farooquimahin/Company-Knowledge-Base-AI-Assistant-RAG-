# Company Knowledge Base AI Assistant (RAG)

Answers questions from company documents and cites its sources.

**Stack:** Python, LangChain, Gemini API, ChromaDB, Streamlit

## How it works
1. Documents (PDF/TXT/MD) are split into overlapping chunks and embedded with a Gemini embedding model.
2. Chunks are stored in a **persistent ChromaDB** collection on disk (`chroma_db/`, or `CHROMA_DIR`).
3. **Multi-turn:** follow-up questions ("what about contractors?") are rewritten into standalone
   search queries using the recent chat history, then the top-k chunks are retrieved.
4. Gemini answers using only those chunks and cites them as [1], [2]...
5. The UI shows each cited file, page and snippet (and what was actually searched for).

## Knowledge base management
- Upload files and click **Add to knowledge base**. Re-uploading a file replaces its old chunks.
- Files placed in `docs/` are indexed automatically on startup if not already indexed.
- Remove a single document, or reset everything, from the sidebar.
- The store is shared by all users of the app (one company knowledge base).

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"  # PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```


**Docker on a VM (truly persistent):**
```bash
docker build -t company-kb .
docker run -d -p 8501:8501 -e GEMINI_API_KEY="key--" -v kb_data:/data company-kb
```
The `kb_data` volume keeps the ChromaDB index across restarts and redeploys.
Platforms with persistent disks (Render, Railway, Fly.io volumes) work the same way:
mount a volume and point `CHROMA_DIR` at it.
