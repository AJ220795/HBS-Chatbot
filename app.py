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
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig, Part, Image

from docx import Document
from pypdf import PdfReader
import cv2
import pytesseract
from PIL import Image as PILImage

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
st.set_page_config(page_title="HBS Help Chatbot", page_icon="🤖", layout="wide")
st.title("HBS Help Chatbot")

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
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_built" not in st.session_state:
    st.session_state.index_built = False

# ---- Credentials via Streamlit Secrets ----
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
    st.error(f"Authentication failed: {e}")
    st.stop()

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
    try:
        img = PILImage.open(io.BytesIO(b)).convert("RGB")
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

# ---- Conversational responses ----
def get_conversational_response(prompt: str) -> str:
    """Handle simple conversational inputs"""
    prompt_lower = prompt.lower().strip()
    
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    farewells = ["bye", "goodbye", "see you", "farewell", "take care"]
    thanks = ["thank you", "thanks", "appreciate it"]
    clarifications = ["i don't understand", "can you elaborate", "explain more", "what do you mean"]
    how_are_you = ["how are you", "how are u", "how's it going", "what's up"]
    
    if any(greeting in prompt_lower for greeting in greetings):
        return "Hello! I'm your HBS Help Chatbot. I can help you with questions about HBS systems, features, and procedures. What would you like to know?"
    
    elif any(farewell in prompt_lower for farewell in farewells):
        return "Goodbye! Feel free to come back anytime if you have more questions about HBS systems."
    
    elif any(thank in prompt_lower for thank in thanks):
        return "You're welcome! I'm here to help with any HBS-related questions you might have."
    
    elif any(clarification in prompt_lower for clarification in clarifications):
        return "I'd be happy to clarify! Could you please be more specific about what you'd like me to explain? For example, you could ask about specific HBS features, procedures, or system functions."
    
    elif any(how in prompt_lower for how in how_are_you):
        return "I'm doing well, thank you! I'm here and ready to help you with any HBS system questions. What can I assist you with today?"
    
    return None  # Not a conversational response

# ---- Pre-load KB files ----
def load_kb_files():
    """Load KB files from the kb directory"""
    kb_files = []
    for ext in [".docx", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        kb_files.extend(KB_DIR.glob(f"*{ext}"))
    return kb_files

def build_index_from_kb_files():
    """Build index from pre-loaded KB files"""
    kb_files = load_kb_files()
    if not kb_files:
        st.error("No KB files found in the kb directory. Please add your documents to the kb folder.")
        return False
    
    texts, corpus = [], []
    
    # Extract text from files
    for file_path in kb_files:
        try:
            with open(file_path, "rb") as f:
                raw = f.read()
            
            ext = file_path.suffix.lower()
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
                
            out_path = EXTRACT_DIR / f"{file_path.stem}.txt"
            out_path.write_text(text, encoding="utf-8")
        except Exception as e:
            st.warning(f"Failed to process {file_path.name}: {e}")

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
        st.error("No text extracted from KB files.")
        return False

    with st.spinner("Building knowledge base index..."):
        embs = embed_texts(texts, st.session_state.project_id, st.session_state.location, st.session_state.creds)
        dim = int(embs.shape[1])
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

    st.session_state.index = index
    st.session_state.corpus = corpus
    return True

# ---- Initialize app ----
def initialize_app():
    """Initialize the app - load index or build from KB files"""
    # Try to load existing index first
    index, corpus = load_index_and_corpus()
    if index is not None and corpus is not None:
        st.session_state.index = index
        st.session_state.corpus = corpus
        st.session_state.index_built = True
        return True
    
    # If no index exists, build from KB files
    success = build_index_from_kb_files()
    if success:
        st.session_state.index_built = True
    return success

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

# ---- Initialize app ----
if not st.session_state.index_built:
    if initialize_app():
        st.success("Knowledge base loaded successfully!")
    else:
        st.error("Failed to load knowledge base. Please check your KB files.")
        st.stop()

# ---- Welcome message ----
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "Hi! How can I help you?"})

# ---- Custom CSS for fixed chat input ----
st.markdown("""
<style>
.stChatInput {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 999 !important;
    background: var(--background-color) !important;
    border-top: 1px solid var(--border-color) !important;
    padding: 1rem !important;
}

.stChatInput > div {
    max-width: 100% !important;
}

.main .block-container {
    padding-bottom: 120px !important;
}
</style>
""", unsafe_allow_html=True)

# ---- Chat UI ----
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message:
            with st.expander("Sources"):
                for h in message["citations"][:3]:
                    src = Path(h["chunk"]["source"]).name
                    snippet = h["chunk"]["text"][:240].replace("\n", " ")
                    st.markdown(f"- {src} (score {h['score']:.3f}): {snippet}...")

# Chat input with file attachment - FIXED POSITION
st.markdown("---")

# Create a container for the input area
input_container = st.container()

with input_container:
    col1, col2, col3 = st.columns([1, 20, 1])
    
    with col1:
        uploaded_image = st.file_uploader(
            "📎", 
            type=["png", "jpg", "jpeg", "webp"], 
            key="image_upload", 
            help="Attach image",
            label_visibility="collapsed"
        )
    
    with col2:
        prompt = st.chat_input("Ask me anything about HBS systems...")
    
    with col3:
        if uploaded_image:
            st.markdown(f"📎 {uploaded_image.name}")

if prompt:
    # Add user message to chat history
    user_content = prompt
    if uploaded_image:
        user_content += f" [Image: {uploaded_image.name}]"
    
    st.session_state.messages.append({"role": "user", "content": user_content})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        if uploaded_image:
            st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                ensure_model_ready()
                
                # Check for conversational responses first
                conversational_response = get_conversational_response(prompt)
                if conversational_response:
                    st.markdown(conversational_response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": conversational_response
                    })
                else:
                    # Handle image + text questions
                    if uploaded_image:
                        # Image-based question
                        hits = retrieve(prompt, k=12)
                        context = build_context(hits) if hits else ""
                        
                        system_instruction = (
                            "You are a helpful HBS systems assistant. Use the provided knowledge base context and attached image to answer. "
                            "If the information is not available, explicitly say you don't know and ask a clarifying question. "
                            "Do NOT use outside knowledge. Do NOT hallucinate. Be concise and helpful."
                        )
                        
                        user_message = (
                            f"{system_instruction}\n\n"
                            f"Context from KB:\n{context}\n\n"
                            f"Question: {prompt}"
                        )
                        
                        # Prepare image
                        image_bytes = uploaded_image.read()
                        image_obj = Image.load_from_file(io.BytesIO(image_bytes))
                        img_part = Part.from_image(image_obj)
                        
                        model = GenerativeModel(st.session_state.gemini_model)
                        resp = model.generate_content([user_message, img_part], generation_config=gen_config)
                        response = resp.text or "I'm sorry, I couldn't generate a response."
                        citations = hits
                        
                    else:
                        # Text-only question
                        hits = retrieve(prompt, k=12)
                        
                        if not context_is_sufficient(hits):
                            response = (
                                "I don't have enough information in the knowledge base to answer that. "
                                "Could you please rephrase your question or ask about something else?"
                            )
                            citations = []
                        else:
                            context = build_context(hits)
                            system_instruction = (
                                "You are a helpful HBS systems assistant. Use ONLY the provided knowledge base context to answer. "
                                "If the information is not in the context, explicitly say you don't know and ask a clarifying question. "
                                "Do NOT use outside knowledge. Do NOT hallucinate. Be concise and helpful."
                            )
                            user_message = (
                                f"{system_instruction}\n\n"
                                f"Context from KB:\n{context}\n\n"
                                f"Question: {prompt}"
                            )
                            model = GenerativeModel(st.session_state.gemini_model)
                            resp = model.generate_content([user_message], generation_config=gen_config)
                            response = resp.text or "I'm sorry, I couldn't generate a response."
                            citations = hits

                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "citations": citations
                    })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ---- Sidebar for admin functions ----
with st.sidebar:
    st.header("Admin")
    
    if st.button("Rebuild Knowledge Base", key="rebuild_btn"):
        if build_index_from_kb_files():
            st.session_state.index_built = True
            st.success("Knowledge base rebuilt successfully!")
            st.rerun()
    
    if st.button("Clear Chat History", key="clear_btn"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you?"}]
        st.rerun()
    
    # Show KB status
    st.subheader("Knowledge Base Status")
    if st.session_state.index_built and st.session_state.corpus:
        st.success(f"✅ Loaded ({len(st.session_state.corpus)} chunks)")
    else:
        st.error("❌ Not loaded")
    
    # Show KB files
    kb_files = load_kb_files()
    st.subheader("KB Files")
    if kb_files:
        for f in kb_files:
            st.text(f"📄 {f.name}")
    else:
        st.warning("No KB files found")
