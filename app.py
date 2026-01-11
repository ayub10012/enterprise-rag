import os
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from typing import List, Optional

# LangChain + Gemini
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

# ================= ENV =================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ================= PATHS =================
APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
VECTOR_DIR = APP_DIR / "vector_index"

UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
VECTOR_DIR.mkdir(exist_ok=True, parents=True)

# ================= MODELS =================
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "models/gemini-2.5-flash"

COLLECTION_NAME = "enterprise_docs"

# ================= CHUNKING =================
CHUNK_SIZE = 900
CHUNK_OVERLAP = 200

# ================= UI =================
st.set_page_config("Enterprise Knowledge Copilot", layout="wide")
st.title("🏢 RAG-Based Enterprise Knowledge Copilot")

st.markdown("""
Ask questions over **enterprise documents** such as:
- Policies
- Technical manuals
- Compliance reports
- Internal documentation

All answers are **grounded, factual, and source-aware**.
""")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Settings")
    persist_db = st.checkbox("Persist Vector Database", value=True)
    force_rebuild = st.checkbox("Force Rebuild Index", value=False)

# ================= FILE UPLOAD =================
uploaded_files = st.file_uploader(
    "📄 Upload Enterprise PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

# ================= HELPERS =================
def save_uploaded_files(files) -> List[Path]:
    saved = []
    for f in files or []:
        path = UPLOAD_DIR / f.name
        with open(path, "wb") as out:
            out.write(f.getvalue())
        saved.append(path)
    return saved


def load_and_chunk(paths: List[Path]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = []
    for p in paths:
        loader = PyPDFLoader(str(p))
        pages = loader.load()
        for d in splitter.split_documents(pages):
            d.metadata["source"] = p.name
            docs.append(d)
    return docs


def build_or_load_vector_db(embedder, docs=None):
    if VECTOR_DIR.exists() and not force_rebuild:
        return Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embedder,
            collection_name=COLLECTION_NAME
        )

    if not docs:
        return None

    vs = Chroma.from_documents(
        docs,
        embedder,
        persist_directory=str(VECTOR_DIR),
        collection_name=COLLECTION_NAME
    )

    if persist_db:
        vs.persist()

    return vs


def retrieve_docs(vs: Optional[Chroma], query: str, k: int = 6):
    if not vs:
        return []
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k}
    ).invoke(query)


def build_context(docs):
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "N/A")
        blocks.append(
            f"[Source: {src}, Page {page}]\n{d.page_content[:800]}"
        )
    return "\n\n---\n\n".join(blocks)


def make_qa_prompt(context, question):
    return f"""
You are an enterprise knowledge assistant.

Answer ONLY using the context.
If not found, say: Information not available.

Context:
{context}

Question:
{question}

Rules:
- Be factual
- Cite sources in square brackets
"""

# ================= EVALUATION =================
def evaluate_rag(answer, docs):
    combined = " ".join([d.page_content.lower() for d in docs])
    answer_words = answer.lower().split()

    supported = [w for w in answer_words if w in combined]
    groundedness = round(len(supported) / max(len(answer_words), 1), 2)

    citation_present = "[" in answer and "]" in answer
    hallucinated = groundedness < 0.4

    retrieval_quality = round(np.mean([len(d.page_content) for d in docs]) / 1000, 2)

    return {
        "Groundedness": groundedness,
        "Citation Coverage": "Yes" if citation_present else "No",
        "Hallucination": "Yes" if hallucinated else "No",
        "Retrieval Quality": retrieval_quality
    }

# ================= MODELS =================
embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)

# ================= INGEST =================
saved_files = save_uploaded_files(uploaded_files)
docs = load_and_chunk(saved_files) if saved_files else []

st.info(f"📄 {len(saved_files)} document(s) uploaded")

with st.spinner("📦 Building / Loading Vector Database..."):
    vector_db = build_or_load_vector_db(embedder, docs)

# ================= QUERY =================
st.markdown("---")
st.subheader("💬 Ask a Question")

user_query = st.text_input("Enter your enterprise-related question")

if st.button("🔍 Get Answer"):
    if not user_query.strip():
        st.warning("Enter a question")
        st.stop()

    if not vector_db:
        st.error("No documents indexed.")
        st.stop()

    start = time.time()
    retrieved = retrieve_docs(vector_db, user_query)
    context = build_context(retrieved)
    response = llm.invoke(make_qa_prompt(context, user_query))
    answer = response.content
    latency = round(time.time() - start, 2)

    metrics = evaluate_rag(answer, retrieved)

    st.markdown("### ✅ Answer")
    st.write(answer)

    st.markdown("### 📊 Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Groundedness", metrics["Groundedness"])
    col2.metric("Citation Coverage", metrics["Citation Coverage"])
    col3.metric("Retrieval Quality", metrics["Retrieval Quality"])
    col4.metric("Latency (sec)", latency)

    if metrics["Hallucination"] == "Yes":
        st.error("⚠️ Possible hallucination detected")
    else:
        st.success("✅ Answer is well grounded")

    with st.expander("📚 Sources Used"):
        for d in retrieved:
            st.markdown(f"**{d.metadata['source']} (Page {d.metadata.get('page')})**")
            st.write(d.page_content[:400])
