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

# ---- Enhanced Conversational responses ----
def get_conversational_response(prompt: str) -> str:
    """Handle simple conversational inputs with better reasoning"""
    prompt_lower = prompt.lower().strip()
    
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    farewells = ["bye", "goodbye", "see you", "farewell", "take care"]
    thanks = ["thank you", "thanks", "appreciate it"]
    clarifications = ["i don't understand", "can you elaborate", "explain more", "what do you mean"]
    how_are_you = ["how are you", "how are u", "how's it going", "what's up"]
    
    if any(greeting in prompt_lower for greeting in greetings):
        return """Hello! I'm your HBS Help Chatbot, specialized in assisting with HBS (Hospitality Business Systems) software. 

I can help you with:
• **System Navigation** - Finding specific functions and features
• **Report Generation** - Creating and understanding various reports (RR, REL, RRR, OER, etc.)
• **Process Guidance** - Step-by-step procedures for common tasks
• **Troubleshooting** - Resolving issues and understanding error messages
• **Feature Explanations** - Understanding what each module does and how to use it

What specific HBS function or process would you like help with today?"""
    
    elif any(farewell in prompt_lower for farewell in farewells):
        return "Goodbye! I'm always here when you need help with HBS systems. Feel free to return anytime you have questions about reports, procedures, or any other HBS functionality."
    
    elif any(thank in prompt_lower for thank in thanks):
        return "You're very welcome! I'm here to make your HBS experience smoother. If you have any other questions about reports, procedures, or system features, just ask!"
    
    elif any(clarification in prompt_lower for clarification in clarifications):
        return """I'd be happy to provide more detailed explanations! To give you the most helpful response, could you specify:

• **What type of information** you're looking for (reports, procedures, features, troubleshooting)
• **Which HBS module** you're working with (Rental Reports, Equipment Lists, etc.)
• **What specific aspect** you'd like me to elaborate on

For example, you could ask:
- "Explain how to generate a Rental Equipment List step by step"
- "What are the different sorting options in Rental Reports?"
- "How do I troubleshoot when a report won't print?"

The more specific your question, the better I can assist you!"""
    
    elif any(how in prompt_lower for how in how_are_you):
        return """I'm functioning perfectly and ready to help! My knowledge base is loaded with comprehensive HBS documentation, so I'm well-equipped to assist with:

• Report generation and configuration
• System navigation and feature explanations  
• Step-by-step procedures
• Troubleshooting common issues
• Understanding different HBS modules and their functions

What HBS-related task can I help you with today?"""
    
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

# ---- Enhanced Retrieval ----
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

# Enhanced generation config for better reasoning
gen_config = GenerationConfig(
    temperature=0.1,  # Lower temperature for more focused reasoning
    top_p=0.8,        # More focused token selection
    max_output_tokens=2048,  # Allow longer, more detailed responses
)

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

# Simple working input - just use the native chat_input
prompt = st.chat_input("Ask me anything about HBS systems...")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

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
                    # Enhanced text-only question with better reasoning
                    hits = retrieve(prompt, k=12)
                    
                    if not context_is_sufficient(hits):
                        response = """I don't have sufficient information in the knowledge base to provide a comprehensive answer to your question.

**What I can suggest:**
• **Reframe your question** - Try asking about specific HBS functions, reports, or procedures
• **Be more specific** - Instead of "how do I...", try "how do I generate a Rental Equipment List?"
• **Check the available modules** - I have documentation for RR, REL, RRR, OER, ROR, and RCR

**Example questions I can help with:**
- "What are the steps to create a Rental Equipment List?"
- "How do I configure sorting options in Rental Reports?"
- "What information is included in an Overdue Equipment Report?"

Could you provide more details about what specific HBS function or process you're trying to understand?"""
                        citations = []
                    else:
                        context = build_context(hits)
                        
                        # Enhanced system instruction with chain of thought
                        system_instruction = """You are an expert HBS (Hospitality Business Systems) assistant with deep knowledge of rental management software. 

**Your approach should be:**
1. **Analyze the question** - Understand what the user is trying to accomplish
2. **Identify relevant information** - Extract key details from the provided context
3. **Provide structured guidance** - Break down complex procedures into clear steps
4. **Explain the reasoning** - Help the user understand WHY each step is important
5. **Offer alternatives** - Suggest different approaches when applicable
6. **Anticipate follow-ups** - Mention related functions or potential issues

**Response format:**
- Start with a brief summary of what you'll explain
- Use clear headings and bullet points for procedures
- Explain the purpose and benefits of each step
- Include any important considerations or prerequisites
- End with related functions or next steps

**Important:** Use ONLY the provided knowledge base context. If information is missing, explicitly state what's not available and suggest how the user might find it. Do NOT use outside knowledge or make assumptions about HBS functionality not documented in the context."""
                        
                        user_message = (
                            f"{system_instruction}\n\n"
                            f"**Knowledge Base Context:**\n{context}\n\n"
                            f"**User Question:** {prompt}\n\n"
                            f"**Instructions:** Provide a comprehensive, well-structured response that helps the user understand not just what to do, but why and how it fits into their broader workflow."
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

# ---- Sidebar for admin functions and image upload ----
with st.sidebar:
    st.header("Attach Image")
    uploaded_image = st.file_uploader(
        "Choose an image", 
        type=["png", "jpg", "jpeg", "webp"], 
        key="image_upload",
        help="Upload an image to ask questions about it"
    )
    
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

# Handle image questions separately
if uploaded_image and st.button("Ask about this image", key="ask_image_btn"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": f"Tell me about this image: {uploaded_image.name}"})
    
    with st.chat_message("user"):
        st.markdown(f"Tell me about this image: {uploaded_image.name}")
        st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing image..."):
            try:
                ensure_model_ready()
                
                # Image-based question
                hits = retrieve("image analysis", k=12)
                context = build_context(hits) if hits else ""
                
                system_instruction = """You are an expert HBS assistant analyzing a screenshot or document image. 

**Your analysis should include:**
1. **Visual identification** - What HBS screen, report, or interface is shown
2. **Functional explanation** - What this screen/report is used for
3. **Key elements** - Important fields, buttons, options visible
4. **User guidance** - How to navigate to this screen or generate this report
5. **Related functions** - What other related screens or reports exist

**Response format:**
- Start with identifying what you see
- Explain the purpose and context
- Break down the key elements and their functions
- Provide navigation guidance
- Suggest related functions or next steps"""
                
                user_message = (
                    f"{system_instruction}\n\n"
                    f"**Knowledge Base Context:**\n{context}\n\n"
                    f"**Image Analysis Request:** Analyze this HBS screenshot/document and explain what it shows, how to access it, and how to use it effectively."
                )
                
                # Save image to temp file first
                temp_image_path = f"/tmp/{uploaded_image.name}"
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_image.getvalue())
                
                # Load image from file path
                image_obj = Image.load_from_file(temp_image_path)
                img_part = Part.from_image(image_obj)
                
                model = GenerativeModel(st.session_state.gemini_model)
                resp = model.generate_content([user_message, img_part], generation_config=gen_config)
                response = resp.text or "I'm sorry, I couldn't generate a response."
                citations = hits
                
                # Clean up temp file
                os.remove(temp_image_path)
                
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
