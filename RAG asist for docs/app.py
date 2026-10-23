import io
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Streamlit Cloud ships an old SQLite that ChromaDB rejects; use pysqlite3 if present.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

st.set_page_config(page_title="Company Knowledge Base Assistant", page_icon="📚", layout="wide")

DEFAULT_CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
DEFAULT_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
PERSIST_DIR = os.getenv("CHROMA_DIR", "chroma_db")  # on disk, survives app restarts
COLLECTION = "company_kb"
DOCS_DIR = Path("docs")
SUPPORTED = {".pdf", ".txt", ".md"}
MAX_HISTORY = 6  # last N chat messages used to understand follow-up questions

REWRITE_PROMPT = """Given the conversation and a follow-up question, rewrite the follow-up as a
single standalone question that can be understood without the conversation.
Keep names, numbers and terms exactly. If it is already standalone, return it unchanged.
Return ONLY the question.

CONVERSATION:
{history}

FOLLOW-UP QUESTION: {question}

STANDALONE QUESTION:"""

ANSWER_PROMPT = """You are a company knowledge base assistant.
Answer the question using ONLY the numbered context passages below.
Cite the passages you used inline like [1] or [2][3].
If the context does not contain the answer, say you could not find it in the documents.
Do not make up policies, numbers, or names.
The recent conversation is only to help you understand the question; every fact must come from the context.

RECENT CONVERSATION:
{history}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# ---------------- helpers ----------------
def get_api_key() -> str | None:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or st.session_state.get("sidebar_key")


def get_store(api_key: str | None, embed_model: str) -> Chroma | None:
    """Open (or create) the persistent ChromaDB collection shared by all sessions."""
    if not api_key:
        return None
    cached = st.session_state.get("store_obj")
    if cached and cached[0] == embed_model:
        return cached[1]
    embeddings = GoogleGenerativeAIEmbeddings(model=embed_model, google_api_key=api_key)
    store = Chroma(collection_name=COLLECTION, embedding_function=embeddings,
                   persist_directory=PERSIST_DIR)
    st.session_state.store_obj = (embed_model, store)
    return store


def indexed_sources(store: Chroma) -> Counter:
    metas = store.get(include=["metadatas"]).get("metadatas") or []
    return Counter(m.get("source", "unknown") for m in metas if m)


def load_file(name: str, data: bytes) -> list[Document]:
    ext = Path(name).suffix.lower()
    docs: list[Document] = []
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": name, "page": i}))
    elif ext in {".txt", ".md"}:
        text = data.decode("utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": name, "page": 1}))
    return docs


def add_files(store: Chroma, files: list[tuple[str, bytes]], chunk_size: int, overlap: int):
    """Add files to the persistent store. Re-uploading a file replaces its old chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    added, skipped = [], []
    for name, data in files:
        docs = load_file(name, data)
        if not docs:
            skipped.append(name)
            continue
        old_ids = store.get(where={"source": name}).get("ids") or []
        if old_ids:
            store.delete(ids=old_ids)
        chunks = splitter.split_documents(docs)
        batch = 32  # small batches + one retry keeps us under free-tier rate limits
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            try:
                store.add_documents(part)
            except Exception:
                time.sleep(10)
                store.add_documents(part)
            time.sleep(0.5)
        added.append((name, len(chunks)))
    return added, skipped


def remove_source(store: Chroma, name: str):
    ids = store.get(where={"source": name}).get("ids") or []
    if ids:
        store.delete(ids=ids)


def message_text(content) -> str:
    """Newer LangChain versions may return a list of content parts."""
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)


def format_history(history: list[dict]) -> str:
    recent = history[-MAX_HISTORY:]
    if not recent:
        return "(none)"
    lines = []
    for m in recent:
        who = "User" if m["role"] == "user" else "Assistant"
        text = m["content"].strip().replace("\n", " ")
        lines.append(f"{who}: {text[:600]}")
    return "\n".join(lines)


def answer_question(store: Chroma, question: str, history: list[dict],
                    api_key: str, model: str, k: int):
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)

    # 1) Turn follow-ups ("what about contractors?") into standalone search queries.
    search_query = question
    if history:
        rewritten = message_text(llm.invoke(
            REWRITE_PROMPT.format(history=format_history(history), question=question)).content).strip()
        if rewritten:
            search_query = rewritten

    # 2) Retrieve and answer with citations.
    hits = store.similarity_search(search_query, k=k)
    context = "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source')}, page {d.metadata.get('page')})\n{d.page_content}"
        for i, d in enumerate(hits, start=1)
    )
    reply = llm.invoke(ANSWER_PROMPT.format(
        history=format_history(history), context=context, question=search_query))
    return message_text(reply.content), hits, search_query


def render_sources(hits: list[Document], search_query: str | None = None, original: str | None = None):
    with st.expander(f"📎 Sources ({len(hits)})"):
        if search_query and original and search_query.strip() != original.strip():
            st.caption(f"Searched for: {search_query}")
        for i, d in enumerate(hits, start=1):
            st.markdown(f"**[{i}] {d.metadata.get('source')}**, page {d.metadata.get('page')}")
            snippet = d.page_content.strip().replace("\n", " ")
            st.caption(snippet[:400] + ("..." if len(snippet) > 400 else ""))


# ---------------- state ----------------
st.session_state.setdefault("messages", [])

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    if get_api_key():
        st.success("API key loaded")
    else:
        st.text_input("Gemini API key", type="password", key="sidebar_key")
    chat_model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
    embed_model = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
    st.caption("If you change the embedding model, use Reset to rebuild the knowledge base.")
    k = st.slider("Passages retrieved (top-k)", 2, 10, 4)
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)

    store = get_store(get_api_key(), embed_model)

    # Index anything in docs/ that is not in the store yet (once per session).
    repo_files = sorted(p for p in DOCS_DIR.glob("*") if p.suffix.lower() in SUPPORTED) if DOCS_DIR.exists() else []
    if store is not None and repo_files and not st.session_state.get("synced"):
        st.session_state.synced = True
        known = set(indexed_sources(store))
        new_files = [(p.name, p.read_bytes()) for p in repo_files if p.name not in known]
        if new_files:
            try:
                with st.spinner(f"Indexing {len(new_files)} file(s) from docs/ ..."):
                    add_files(store, new_files, chunk_size, overlap)
            except Exception as e:
                st.error(f"Indexing docs/ failed: {e}")

    st.divider()
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF, TXT or MD", type=["pdf", "txt", "md"],
                               accept_multiple_files=True)
    if st.button("Add to knowledge base", type="primary", use_container_width=True):
        if store is None:
            st.error("Please provide a Gemini API key.")
        elif not uploads:
            st.error("Upload at least one document.")
        else:
            try:
                with st.spinner("Reading, chunking and embedding..."):
                    added, skipped = add_files(store, [(u.name, u.getvalue()) for u in uploads],
                                               chunk_size, overlap)
                for name, n in added:
                    st.success(f"{name}: {n} chunks")
                for name in skipped:
                    st.warning(f"{name}: no readable text (scanned PDF?)")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    if store is not None:
        sources = indexed_sources(store)
        st.subheader(f"Indexed documents ({len(sources)})")
        if sources:
            for name, n in sorted(sources.items()):
                st.caption(f"📄 {name}: {n} chunks")
            to_remove = st.selectbox("Remove a document", ["(select)"] + sorted(sources))
            if st.button("Remove selected", use_container_width=True) and to_remove != "(select)":
                remove_source(store, to_remove)
                st.rerun()
        else:
            st.caption("Nothing indexed yet.")

        with st.expander("Danger zone"):
            if st.button("Reset entire knowledge base", use_container_width=True):
                store.delete_collection()
                st.session_state.pop("store_obj", None)
                st.session_state.messages = []
                st.rerun()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- main ----------------
st.title("📚 Company Knowledge Base Assistant")
st.caption("Ask questions about your company documents. Follow-up questions are understood in context, and answers cite their sources.")

has_docs = store is not None and bool(indexed_sources(store))
if store is None:
    st.info("Add your Gemini API key in the sidebar to begin.")
elif not has_docs:
    st.info("The knowledge base is empty. Upload documents in the sidebar, or put files in the `docs/` folder.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            render_sources(m["hits"], m.get("search_query"), m.get("question"))

question = st.chat_input("Ask a question about your documents...", disabled=not has_docs)
if question:
    history = list(st.session_state.messages)  # conversation before this question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                text, hits, search_query = answer_question(store, question, history,
                                                           get_api_key(), chat_model, k)
            st.markdown(text)
            render_sources(hits, search_query, question)
            st.session_state.messages.append({"role": "assistant", "content": text, "hits": hits,
                                              "search_query": search_query, "question": question})
        except Exception as e:
            st.error(f"Could not answer: {e}")
