import os
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
CHAT_MODEL = "gemini-2.5-flash"

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
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k}
    )
    return retriever.invoke(query)


def build_context(docs):
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "N/A")
        blocks.append(
            f"[Source: {src}, Page {page}]\n{d.page_content[:1000]}"
        )
    return "\n\n---\n\n".join(blocks)


def make_qa_prompt(context, question):
    return f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the context provided.
If the answer is not present, respond with:
"Information not available in the provided documents."

Context:
{context}

Question:
{question}

Rules:
- Be concise and factual
- Do not hallucinate
- Always cite sources in square brackets
"""

# ================= MODELS INIT =================
try:
    embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)
except Exception as e:
    st.error(f"Model initialization failed: {e}")
    st.stop()

# ================= INGESTION =================
saved_files = save_uploaded_files(uploaded_files)
docs = load_and_chunk(saved_files) if saved_files else []

st.info(f"📄 {len(saved_files)} document(s) uploaded")

with st.spinner("📦 Building / Loading Vector Database..."):
    vector_db = build_or_load_vector_db(embedder, docs)

# ================= QUERY =================
st.markdown("---")
st.subheader("💬 Ask a Question")

user_query = st.text_input(
    "Enter your enterprise-related question",
    placeholder="e.g. What is the data retention policy?"
)

if st.button("🔍 Get Answer"):
    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not vector_db:
        st.error("No documents indexed.")
        st.stop()

    with st.spinner("🧠 Retrieving context and generating answer..."):
        retrieved = retrieve_docs(vector_db, user_query)
        context = build_context(retrieved)

        prompt = make_qa_prompt(context, user_query)
        response = llm.invoke(prompt)
        answer = getattr(response, "content", response)

        st.markdown("### ✅ Answer")
        st.write(answer)

        with st.expander("📚 Sources Used"):
            for i, d in enumerate(retrieved, 1):
                st.markdown(
                    f"**{i}. {d.metadata.get('source')} (Page {d.metadata.get('page')})**"
                )
                st.write(d.page_content[:500])
