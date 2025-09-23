# app.py
import os
import io
import json
import re
from pathlib import Path
from typing import List, Dict

import streamlit as st
import numpy as np
import faiss

from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai import init as vertexai_init
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig

from docx import Document
from pypdf import PdfReader
import cv2
import pytesseract
from PIL import Image

# ---- App constants ----
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
KB_DIR = DATA_DIR / "kb"
EXTRACT_DIR = DATA_DIR / "kb_extracted"
INDEX_PATH = DATA_DIR / "faiss.index"
CORPUS_PATH = DATA_DIR / "corpus.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview",
    "gemini-2.5-flash-lite-001",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-001",
]

DEFAULT_LOCATION = "us-central1"

# ---- Streamlit page ----
st.set_page_config(page_title="RAG Chatbot (Vertex AI + FAISS)", page_icon="🤖", layout="wide")
st.title("RAG Chatbot (Vertex AI + FAISS + Streamlit)")

# ---- Session defaults ----
if "creds" not in st.session_state:
    st.session_state.creds = None
if "project_id" not in st.session_state:
    st.session_state.project_id = None
if "location" not in st.session_state:
    st.session_state.location = DEFAULT_LOCATION
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None
if "index" not in st.session_state:
    st.session_state.index = None
if "corpus" not in st.session_state:
    st.session_state.corpus = None
if "history" not in st.session_state:
    st.session_state.history = []

# ---- Credentials via Streamlit Secrets (preferred on Streamlit Cloud) ----
try:
    if "google" in st.secrets:
        sa_info = json.loads(st.secrets["google"]["credentials_json"])
        creds = service_account.Credentials.from_service_account_info(sa_info)
        project_id = st.secrets["google"]["project"]
        location_secret = st.secrets["google"].get("location", DEFAULT_LOCATION)

        aiplatform.init(project=project_id, location=location_secret, credentials=creds)
        vertexai_init(project=project_id, location=location_secret, credentials=creds)

        st.session_state.creds = creds
        st.session_state.project_id = project_id
        st.session_state.location = location_secret
except Exception as e:
    st.warning(f"Secrets-based auth not configured or failed: {e}")

# ---- Sidebar ----
with st.sidebar:
    st.header("1) Credentials")
    if st.session_state.creds:
        st.success(f"Authenticated. Project: {st.session_state.project_id} | Region: {st.session_state.location}")
        st.caption("Using Streamlit Secrets.")
    else:
        st.caption("Optional local dev fallback (not for Streamlit Cloud):")
        sa_file = st.file_uploader("Upload Vertex AI service account JSON", type=["json"], key="sa_uploader")
        project_override = st.text_input("Project ID (optional)", key="project_override")
        location_input = st.text_input("Location", value=st.session_state.location, key="location_input")
        apply_creds = st.button("Use Uploaded Credentials", key="apply_creds")
        if apply_creds and sa_file is not None:
            try:
                sa_bytes = sa_file.read()
                sa_info = json.loads(sa_bytes.decode("utf-8"))
                creds = service_account.Credentials.from_service_account_info(sa_info)
                project_id = (project_override or sa_info.get("project_id") or "").strip()
                if not project_id:
                    st.error("Project ID not found. Enter it or include it in the JSON.")
                else:
                    aiplatform.init(project=project_id, location=location_input, credentials=creds)
                    vertexai_init(project=project_id, location=location_input, credentials=creds)
                    st.session_state.creds = creds
                    st.session_state.project_id = project_id
                    st.session_state.location = location_input
                    st.success(f"Authenticated. Project: {project_id} | Region: {location_input}")
            except Exception as e:
                st.error(f"Failed to load credentials: {e}")

    st.header("2) Knowledge Base")
    kb_files = st.file_uploader(
        "Upload KB files (.docx, .pdf, .png/.jpg). You can multi-select",
        type=["docx", "pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        accept_multiple_files=True,
        key="kb_uploader"
    )
    do_build = st.button("Build / Rebuild Index", key="build_btn")

    st.header("3) Persistence")
    do_load = st.button("Load Existing Index", key="load_btn")
    do_save = st.button("Save Current Index", key="save_btn")

# ---- Utilities ----
def split_into_sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def chunk_text(text: str, max_tokens: int = 500, overlap_sentences: int = 1) -> List[str]:
    sents = split_into_sentences(text)
    chunks, buf, token_est = [], [], 0
    for s in sents:
        s_tokens = max(1, len(s) // 4)
        if token_est + s_tokens > max_tokens and buf:
            chunks.append(" ".join(buf))
            buf = buf[-overlap_sentences:] if overlap_sentences > 0 else []
            token_est = sum(max(1, len(x)//4) for x in buf)
        buf.append(s)
        token_est += s_tokens
    if buf:
        chunks.append(" ".join(buf))
    return chunks

def extract_text_from_docx_bytes(b: bytes) -> str:
    f = io.BytesIO(b)
    doc = Document(f)
    out = []
    for para in doc.paragraphs:
        t = (para.text or "").strip()
        if t:
            out.append(t)
    return "\n".join(out)

def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        f = io.BytesIO(b)
        reader = PdfReader(f)
        parts = []
        for page in reader.pages:
            t = (page.extract_text() or "").strip()
            if t:
                parts.append(t)
        return "\n\n".join(parts)
    except Exception:
        return ""

def extract_text_from_image_bytes(b: bytes) -> str:
    # Note: Streamlit Cloud does not provide system Tesseract. This will return "" there.
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return (pytesseract.image_to_string(gray) or "").strip()
    except Exception:
        return ""

def embed_texts(texts: List[str], project_id: str, location: str, creds) -> np.ndarray:
    vertexai_init(project=project_id, location=location, credentials=creds)
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    vecs, bs = [], 32
    for i in range(0, len(texts), bs):
        embs = model.get_embeddings(texts[i:i+bs])
        vecs.extend([e.values for e in embs])
    return np.array(vecs, dtype="float32")

def pick_available_model(creds, project_id: str, location: str) -> str:
    vertexai_init(project=project_id, location=location, credentials=creds)
    gen_cfg = GenerationConfig(max_output_tokens=8)
    for mid in CANDIDATE_MODELS:
        try:
            m = GenerativeModel(mid)
            r = m.generate_content(["ok"], generation_config=gen_cfg)
            if getattr(r, "text", None):
                return mid
        except Exception:
            continue
    raise RuntimeError("No Gemini Flash Lite model available in this region/account.")

def save_index_and_corpus(index, corpus: List[Dict]):
    faiss.write_index(index, str(INDEX_PATH))
    CORPUS_PATH.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

def load_index_and_corpus():
    if INDEX_PATH.exists() and CORPUS_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        return index, corpus
    return None, None

def build_context(hits: List[Dict], max_chars: int = 6000) -> str:
    out, used = [], 0
    for h in hits:
        t = h["chunk"]["text"]
        src = h["chunk"]["source"]
        block = f"[Source: {src}]\n{t}\n"
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return "\n".join(out)

def embed_query(q: str, project_id: str, location: str, creds) -> np.ndarray:
    vertexai_init(project=project_id, location=location, credentials=creds)
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    embs = model.get_embeddings([q])
    return np.array([embs[0].values], dtype="float32")

# ---- Build index ----
def build_index_from_uploads(files):
    if not files:
        st.warning("Upload some KB files first.")
        return
    texts, corpus = [], []
    # Save original files to disk and extract text
    for f in files:
        try:
            raw = f.read()
            dest = KB_DIR / f.name
            dest.write_bytes(raw)
            ext = dest.suffix.lower()
            text = ""
            if ext == ".docx":
                text = extract_text_from_docx_bytes(raw)
            elif ext == ".pdf":
                text = extract_text_from_pdf_bytes(raw)
            elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
                text = extract_text_from_image_bytes(raw)
            else:
                continue
            text = (text or "").strip()
            if not text:
                continue
            out_path = EXTRACT_DIR / f"{dest.stem}.txt"
            out_path.write_text(text, encoding="utf-8")
        except Exception as e:
            st.warning(f"Failed to process {f.name}: {e}")

    # Build corpus from extracted
    docs = []
    for p in sorted(EXTRACT_DIR.glob("*.txt")):
        try:
            t = p.read_text(encoding="utf-8").strip()
        except Exception:
            t = ""
        if t:
            docs.append({"source": str(p), "text": t})

    for d in docs:
        for i, ch in enumerate(chunk_text(d["text"], max_tokens=500, overlap_sentences=1)):
            corpus.append({"id": f'{d["source"]}#{i}', "text": ch, "source": d["source"]})
            texts.append(ch)

    if not texts:
        st.error("No text extracted. Ensure your files contain readable text.")
        return

    with st.spinner("Embedding texts and building FAISS index..."):
        embs = embed_texts(texts, st.session_state.project_id, st.session_state.location, st.session_state.creds)
        dim = int(embs.shape[1])
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

    st.session_state.index = index
    st.session_state.corpus = corpus
    st.success(f"Index built with {len(corpus)} chunks.")

# ---- Button actions ----
if do_build:
    if st.session_state.creds is None or st.session_state.project_id is None:
        st.error("Configure credentials first (use Secrets on Streamlit Cloud).")
    else:
        build_index_from_uploads(kb_files)

if do_load:
    idx, corp = load_index_and_corpus()
    if idx is not None:
        st.session_state.index = idx
        st.session_state.corpus = corp
        st.success(f"Loaded existing index with {len(corp)} chunks.")
    else:
        st.warning("No saved index found. Build first.")

if do_save:
    if st.session_state.index is not None and st.session_state.corpus is not None:
        save_index_and_corpus(st.session_state.index, st.session_state.corpus)
        st.success("Index and corpus saved.")
    else:
        st.warning("Nothing to save. Build an index first.")

# ---- Model selection ----
def ensure_model_ready():
    if st.session_state.gemini_model is None:
        st.session_state.gemini_model = pick_available_model(
            st.session_state.creds, st.session_state.project_id, st.session_state.location
        )

# ---- Retrieval ----
MIN_SCORE = 0.25
def retrieve(query: str, k: int = 12):
    if st.session_state.index is None or st.session_state.corpus is None:
        return []
    q = embed_query(query, st.session_state.project_id, st.session_state.location, st.session_state.creds).astype("float32")
    faiss.normalize_L2(q)
    D, I = st.session_state.index.search(q, k)
    raw = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        raw.append({"score": float(score), "chunk": st.session_state.corpus[idx]})
    hits = [h for h in raw if h["score"] >= MIN_SCORE]
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits

def context_is_sufficient(hits, min_citations=2):
    high = [h for h in hits if h["score"] >= MIN_SCORE]
    return len(high) >= min_citations

gen_config = GenerationConfig(temperature=0.3, top_p=0.9, max_output_tokens=1024)

# ---- Chat UI ----
st.subheader("Chat")
query = st.text_input("Ask a question (answers come strictly from your KB)", key="query_input")
col1, col2 = st.columns([1,1])
with col1:
    ask = st.button("Ask", key="ask_btn")
with col2:
    clear = st.button("Clear conversation", key="clear_btn")

if clear:
    st.session_state.history = []

if ask:
    if st.session_state.creds is None or st.session_state.index is None:
        st.error("Configure credentials and build/load the index first.")
    elif not query.strip():
        st.warning("Enter a question.")
    else:
        try:
            ensure_model_ready()
            hits = retrieve(query, k=12)
            if not context_is_sufficient(hits):
                answer = (
                    "I don't have enough information in the provided context to answer that. "
                    "Please refine your question or upload more relevant documents."
                )
                st.session_state.history.append({"user": query, "assistant": answer, "hits": []})
            else:
                context = build_context(hits)
                system_instruction = (
                    "You are a helpful assistant. Use ONLY the provided KB context to answer. "
                    "If the information is not in the context, explicitly say you don't know and ask a clarifying question. "
                    "Do NOT use outside knowledge. Do NOT hallucinate."
                )
                user_message = (
                    f"{system_instruction}\n\n"
                    f"Context from KB:\n{context}\n\n"
                    f"Question: {query}"
                )
                model = GenerativeModel(st.session_state.gemini_model)
                resp = model.generate_content([user_message], generation_config=gen_config)
                answer = resp.text or ""
                st.session_state.history.append({"user": query, "assistant": answer, "hits": hits})
        except Exception as e:
            st.error(f"Error during generation: {e}")

# ---- Render conversation and citations ----
for turn in st.session_state.history[-8:]:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Assistant:** {turn['assistant']}")
    hits = turn.get("hits", [])
    if hits:
        with st.expander("Citations"):
            for h in hits[:3]:
                src = Path(h["chunk"]["source"]).name
                snippet = h["chunk"]["text"][:240].replace("\n", " ")
                st.markdown(f"- {src} (score {h['score']:.3f}): {snippet}...")
