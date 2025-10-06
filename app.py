import os
import io
import re
import json
import base64
from pathlib import Path
from typing import List, Dict

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import faiss

from google.oauth2 import service_account
from vertexai import init as vertexai_init
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig, Part

from docx import Document
from pypdf import PdfReader
import cv2
import pytesseract
from PIL import Image as PILImage

# -----------------------
# App constants & paths
# -----------------------
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
KB_DIR = APP_DIR / "kb"
INDEX_PATH = DATA_DIR / "faiss.index"
CORPUS_PATH = DATA_DIR / "corpus.json"
DEFAULT_LOCATION = "us-central1"

DATA_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash-lite"]

# -----------------------
# Small utilities
# -----------------------
def split_into_sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def chunk_text(text: str, max_tokens: int = 800, overlap_sentences: int = 2) -> List[str]:
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

# -----------------------
# OCR report parsing (kept minimal)
# -----------------------
def parse_report_data_from_ocr(ocr_text: str, filename: str) -> List[Dict]:
    """
    Very light example: recognize 'overdue' or 'outbound' as hints.
    Extend as needed with your full parsing rules.
    """
    data = []
    txt = ocr_text.lower()
    if "overdue" in txt:
        data.append({"report_type": "Overdue Equipment Report", "source": filename})
    if "outbound" in txt:
        data.append({"report_type": "Rental Outbound Report", "source": filename})
    return data

# -----------------------
# Embeddings + FAISS
# -----------------------
def embed_texts(texts: List[str], project_id: str, location: str, credentials):
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        embeddings = model.get_embeddings(texts)
        return np.array([e.values for e in embeddings]).astype(np.float32)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return np.array([])

def build_faiss_index(corpus: List[Dict], project_id: str, location: str, credentials):
    if not corpus:
        return None, []
    texts = [item["text"] for item in corpus]
    embs = embed_texts(texts, project_id, location, credentials)
    if embs.size == 0:
        return None, []
    dim = embs.shape[1]
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index, corpus

def expand_query(query: str) -> str:
    ql = query.lower()
    if "overdue" in ql:
        return f"{query} overdue equipment report rental"
    if "outbound" in ql:
        return f"{query} outbound report rental equipment"
    if "equipment" in ql:
        return f"{query} equipment list rental"
    if "customer" in ql:
        return f"{query} customer contract phone"
    if "stock" in ql:
        return f"{query} stock number equipment"
    if "serial" in ql:
        return f"{query} serial number equipment"
    return query

def search_index(query: str, index, corpus: List[Dict], project_id: str, location: str, credentials, k: int = 2, min_similarity: float = 0.5) -> List[Dict]:
    if index is None or not corpus:
        return []
    try:
        expanded = expand_query(query)
        qemb = embed_texts([expanded], project_id, location, credentials)
        if qemb.size == 0:
            return []
        faiss.normalize_L2(qemb)
        scores, indices = index.search(qemb, min(10, len(corpus)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(corpus) and score >= min_similarity:
                results.append({**corpus[idx], "similarity_score": float(score)})
                if len(results) >= k:
                    break
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def load_index_and_corpus():
    try:
        if INDEX_PATH.exists() and CORPUS_PATH.exists():
            idx = faiss.read_index(str(INDEX_PATH))
            with open(CORPUS_PATH, "r") as f:
                corpus = json.load(f)
            return idx, corpus
    except Exception as e:
        st.warning(f"Could not read saved index/corpus: {e}")
    return None, []

def save_index_and_corpus(index, corpus: List[Dict]):
    try:
        if index is not None:
            faiss.write_index(index, str(INDEX_PATH))
        with open(CORPUS_PATH, "w") as f:
            json.dump(corpus, f, indent=2)
    except Exception as e:
        st.error(f"Error saving index/corpus: {e}")

def process_kb_files() -> List[Dict]:
    corpus = []
    if not KB_DIR.exists():
        return corpus
    for p in KB_DIR.iterdir():
        if not p.is_file():
            continue
        try:
            if p.suffix.lower() == ".docx":
                text = extract_text_from_docx_bytes(p.read_bytes())
                if text.strip():
                    for i, ch in enumerate(chunk_text(text)):
                        corpus.append({"text": ch, "source": p.name, "chunk_id": i, "file_type": p.suffix.lower()})
            elif p.suffix.lower() == ".pdf":
                text = extract_text_from_pdf_bytes(p.read_bytes())
                if text.strip():
                    for i, ch in enumerate(chunk_text(text)):
                        corpus.append({"text": ch, "source": p.name, "chunk_id": i, "file_type": p.suffix.lower()})
            elif p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
                ocr_text = extract_text_from_image_bytes(p.read_bytes())
                if ocr_text.strip():
                    for i, ch in enumerate(chunk_text(ocr_text)):
                        corpus.append({"text": ch, "source": p.name, "chunk_id": i, "file_type": p.suffix.lower(), "content_type": "ocr_text"})
                    for item in parse_report_data_from_ocr(ocr_text, p.name):
                        corpus.append({
                            "text": f"Report: {item.get('report_type','Unknown')} Source: {p.name}",
                            "source": p.name, "chunk_id": len(corpus), "file_type": p.suffix.lower(),
                            "content_type": "structured_data", "structured_data": item
                        })
        except Exception as e:
            st.warning(f"Failed to process {p.name}: {e}")
    return corpus

# -----------------------
# Conversational helpers
# -----------------------
def get_conversational_response(q: str) -> str | None:
    ql = q.lower().strip()
    if any(g in ql for g in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]):
        return "Hi! How can I help you?"
    if any(f in ql for f in ["bye", "goodbye", "see you", "farewell", "thanks", "thank you", "ttyl"]):
        return "Goodbye! Feel free to come back anytime if you have more questions about HBS systems."
    if any(c in ql for c in ["what's up", "how are you", "how's it going", "what's new"]):
        return "I'm doing well, thanks! What can I help you with in HBS today?"
    return None

def get_conversation_context(messages: List[Dict], max_context: int = 3) -> str:
    if len(messages) < 2:
        return ""
    recent = messages[-max_context*2:]
    parts = []
    for m in recent:
        if m["role"] == "user":
            parts.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"Assistant: {m['content'].split('\\n\\n')[0]}")
    return "\n".join(parts)

def classify_user_intent(query: str, conversation_context: str, model_name: str, project_id: str, location: str, credentials) -> Dict:
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        prompt = f"""Analyze the user's query and classify their intent.

CONTEXT:
{conversation_context}

QUERY: {query}

Return ONLY JSON:
{{
  "intent": "troubleshooting|clarification|alternative|new_question|conversational",
  "confidence": 0.92,
  "reasoning": "short reason"
}}"""
        resp = model.generate_content(prompt, generation_config=GenerationConfig(temperature=0.1, max_output_tokens=200))
        if resp.text:
            try:
                return json.loads(resp.text.strip())
            except Exception:
                return {"intent": "new_question", "confidence": 0.5, "reasoning": "parse error"}
    except Exception as e:
        return {"intent": "new_question", "confidence": 0.3, "reasoning": f"error: {e}"}
    return {"intent": "new_question", "confidence": 0.5, "reasoning": "no response"}

def generate_response(query: str, context_chunks: List[Dict], model_name: str, project_id: str, location: str, credentials, conversation_context: str = "", user_intent: Dict | None = None) -> str:
    if not context_chunks:
        conv = get_conversational_response(query)
        if conv:
            return conv
        return "I don't have information about that in my knowledge base. Try asking about HBS reports, procedures, or system features."
    context_text = "\n\n".join([f"Source: {c['source']}\nContent: {c['text']}" for c in context_chunks])
    intent_section = ""
    if user_intent and "intent" in user_intent:
        intent_section = f"\nUSER INTENT: {user_intent['intent']} ({user_intent.get('confidence',0):.2f}) — {user_intent.get('reasoning','')}\n"
    sys_prompt = f"""You are an HBS assistant.
{intent_section}
CONTEXT:
{context_text}

QUESTION: {query}

Rules:
- Answer ONLY using context above if relevant.
- If insufficient, say so briefly and suggest what to ask next.
- Be precise and list steps/options where applicable.

Answer:"""
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        resp = model.generate_content(sys_prompt, generation_config=GenerationConfig(temperature=0.1, max_output_tokens=1024))
        return resp.text or "I couldn't generate a response. Please rephrase your question."
    except Exception as e:
        return f"Error generating response: {e}"

def generate_image_response(query: str, image_bytes: bytes, model_name: str, project_id: str, location: str, credentials, mime_type: str = "image/jpeg") -> str:
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        image_part = Part.from_data(image_bytes, mime_type=mime_type)
        prompt = f"""Analyze this image for info relevant to HBS. User question: {query}"""
        resp = model.generate_content([prompt, image_part], generation_config=GenerationConfig(temperature=0.1, max_output_tokens=800))
        return resp.text or "I couldn't analyze that image."
    except Exception as e:
        return f"Error analyzing image: {e}"

# -----------------------
# Streamlit app (Custom HTML UI)
# -----------------------
def main():
    st.set_page_config(page_title="HBS Help Chatbot", page_icon="🤖", layout="wide")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []  # We'll render the first greeting directly in HTML, not in state
    if "index" not in st.session_state:
        st.session_state.index = None
    if "corpus" not in st.session_state:
        st.session_state.corpus = []
    if "creds" not in st.session_state:
        st.session_state.creds = None
    if "project_id" not in st.session_state:
        st.session_state.project_id = None
    if "location" not in st.session_state:
        st.session_state.location = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = CANDIDATE_MODELS[0]
    if "kb_loaded" not in st.session_state:
        st.session_state.kb_loaded = False

    # Credentials
    try:
        sa_info = json.loads(st.secrets["google"]["credentials_json"])
        st.session_state.creds = service_account.Credentials.from_service_account_info(sa_info)
        st.session_state.project_id = st.secrets["google"]["project"]
        st.session_state.location = st.secrets["google"].get("location", DEFAULT_LOCATION)
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        st.stop()

    # KB init
    @st.cache_resource
    def initialize_app():
        idx, corp = load_index_and_corpus()
        if idx is not None and corp:
            return idx, corp, True
        corp = process_kb_files()
        if not corp:
            return None, [], False
        idx, corp = build_faiss_index(corp, st.session_state.project_id, st.session_state.location, st.session_state.creds)
        if idx is not None:
            save_index_and_corpus(idx, corp)
            return idx, corp, True
        return None, [], False

    if not st.session_state.kb_loaded:
        with st.spinner("Loading knowledge base..."):
            idx, corp, loaded = initialize_app()
            st.session_state.index = idx
            st.session_state.corpus = corp
            st.session_state.kb_loaded = loaded

    # Sidebar
    with st.sidebar:
        st.header("HBS Help Chatbot")
        if st.session_state.kb_loaded:
            st.success(f"✅ Knowledge base loaded ({len(st.session_state.corpus)} chunks)")
        else:
            st.error("❌ Knowledge base not loaded")

        st.subheader("Model Settings")
        st.session_state.model_name = st.selectbox("Select Model", CANDIDATE_MODELS, index=0, key="model_select")

        if st.button("🔄 Rebuild Index", key="rebuild_btn"):
            with st.spinner("Rebuilding index..."):
                corp = process_kb_files()
                if corp:
                    idx, corp = build_faiss_index(corp, st.session_state.project_id, st.session_state.location, st.session_state.creds)
                    if idx is not None:
                        save_index_and_corpus(idx, corp)
                        st.session_state.index = idx
                        st.session_state.corpus = corp
                        st.session_state.kb_loaded = True
                        st.success("Index rebuilt successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to build index")
                else:
                    st.error("No KB files found")

        if st.button("🗑️ Clear Conversation", key="clear_btn"):
            st.session_state.messages = []
            st.rerun()

    # -------- Build custom HTML (UI you requested) --------
    # Render existing messages inside the HTML thread
    msgs_html = []
    # fixed greeting at top (not stored in session to avoid ordering weirdness)
    msgs_html.append('<div class="message message-assistant">Hi! How can I help you?</div>')
    for m in st.session_state.messages:
        role_class = "message-user" if m["role"] == "user" else "message-assistant"
        msgs_html.append(f'<div class="message {role_class}">{m["content"]}</div>')
        if m.get("sources"):
            src_bits = []
            for s in m["sources"][:2]:
                nm = s.get("source", "Unknown")
                sim = float(s.get("similarity_score", 0.0))
                src_bits.append(f'📄 {nm} (similarity: {sim:.3f})')
            if src_bits:
                msgs_html.append(f'<div class="sources"><strong>Sources:</strong><br>{"<br>".join(src_bits)}</div>')

    chat_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body {{ margin:0; padding:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; height:100vh; overflow:hidden; }}
  .chat-container {{ display:flex; flex-direction:column; height:100vh; background:#0f172a; }}
  .chat-header {{ background:#0f172a; padding:1.25rem 1.5rem; border-bottom:1px solid #1f2937; color:#fff; position:sticky; top:0; z-index:10; }}
  .chat-header h1 {{ margin:0; font-size:1.75rem; }}
  .chat-messages {{ flex:1; overflow-y:auto; padding:1rem; display:flex; flex-direction:column; gap:1rem; }}
  .message {{ max-width:70%; padding:12px 16px; border-radius:18px; word-wrap:break-word; box-shadow:0 2px 4px rgba(0,0,0,0.2); }}
  .message-user {{ background:#2563eb; color:#fff; margin-left:auto; border-radius:18px 18px 5px 18px; }}
  .message-assistant {{ background:#111827; color:#e5e7eb; margin-right:auto; border:1px solid #1f2937; border-radius:18px 18px 18px 5px; }}
  .sources {{ background:#0b253f; color:#dbeafe; padding:8px 12px; border-radius:8px; margin-top:4px; font-size:0.9em; border-left:3px solid #3b82f6; }}
  .chat-input-container {{ background:#0f172a; padding:1rem; border-top:1px solid #1f2937; display:flex; gap:0.75rem; align-items:center; position:sticky; bottom:0; }}
  .chat-input {{ flex:1; padding:12px 16px; border:2px solid #1f2937; border-radius:25px; font-size:16px; outline:none; background:#111827; color:#e5e7eb; }}
  .chat-input:focus {{ border-color:#2563eb; box-shadow:0 0 0 3px rgba(37,99,235,0.2); }}
  .send-button {{ background:#2563eb; color:white; border:none; padding:12px 20px; border-radius:25px; cursor:pointer; font-size:16px; }}
  .send-button:hover {{ background:#1d4ed8; }}
  .upload-button {{ background:#374151; color:white; border:none; padding:12px; border-radius:50%; cursor:pointer; width:44px; height:44px; display:flex; align-items:center; justify-content:center; }}
</style>
</head>
<body>
  <div class="chat-container">
    <div class="chat-header">
      <h1>HBS Help Chatbot</h1>
    </div>
    <div id="chatMessages" class="chat-messages">
      {"".join(msgs_html)}
    </div>
    <div class="chat-input-container">
      <input type="text" class="chat-input" id="chatInput" placeholder="Ask me anything about HBS systems..." />
      <button class="upload-button" id="uploadBtn" title="Upload image">📷</button>
      <input type="file" id="fileInput" accept="image/*" style="display:none;" />
      <button class="send-button" id="sendBtn">Send</button>
    </div>
  </div>

<script>
  const chatInput = document.getElementById('chatInput');
  const chatMessages = document.getElementById('chatMessages');
  const sendBtn = document.getElementById('sendBtn');
  const uploadBtn = document.getElementById('uploadBtn');
  const fileInput = document.getElementById('fileInput');

  function scrollBottom() {{
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }}

  function addMessage(text, role) {{
    const div = document.createElement('div');
    div.className = 'message ' + (role === 'user' ? 'message-user' : 'message-assistant');
    div.textContent = text;
    chatMessages.appendChild(div);
    scrollBottom();
  }}

  function sendToStreamlit(value) {{
    // Streamlit Components protocol:
    window.parent.postMessage({{
      isStreamlitMessage: true,
      type: "streamlit:setComponentValue",
      value: value
    }}, "*");
  }}

  function sendText() {{
    const msg = chatInput.value.trim();
    if (!msg) return;
    addMessage(msg, 'user');
    chatInput.value = '';
    sendToStreamlit({{kind: "text", text: msg}});
  }}

  sendBtn.addEventListener('click', sendText);
  chatInput.addEventListener('keypress', (e) => {{
    if (e.key === 'Enter') sendText();
  }});

  uploadBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {{
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {{
      // Send as data URL
      sendToStreamlit({{kind: "image", name: file.name, mime: file.type || "image/jpeg", data: reader.result}});
    }};
    reader.readAsDataURL(file);
  }});

  window.addEventListener('load', scrollBottom);
</script>
</body>
</html>
"""
    # Render the component and catch events from JS
    evt = components.html(chat_html, height=600, key="chat_ui")

    # evt can be None, or a dict we sent from JS: {"kind": "text" | "image", ...}
    if evt:
        try:
            if isinstance(evt, str):
                # Backward-compat if only a raw string is sent
                user_message = evt
                st.session_state.messages.append({"role": "user", "content": user_message})
            elif isinstance(evt, dict):
                kind = evt.get("kind", "text")
                if kind == "text":
                    user_message = (evt.get("text") or "").strip()
                    if user_message:
                        st.session_state.messages.append({"role": "user", "content": user_message})

                        # Generate a response
                        conversation_context = get_conversation_context(st.session_state.messages)
                        user_intent = classify_user_intent(
                            user_message, conversation_context,
                            st.session_state.model_name,
                            st.session_state.project_id,
                            st.session_state.location,
                            st.session_state.creds
                        ) if conversation_context else None

                        ctx = search_index(
                            user_message,
                            st.session_state.index,
                            st.session_state.corpus,
                            st.session_state.project_id,
                            st.session_state.location,
                            st.session_state.creds,
                            k=2, min_similarity=0.5
                        )
                        answer = generate_response(
                            user_message, ctx,
                            st.session_state.model_name,
                            st.session_state.project_id,
                            st.session_state.location,
                            st.session_state.creds,
                            conversation_context, user_intent
                        )
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": ctx})
                        st.rerun()

                elif kind == "image":
                    # evt["data"] is a data URL: "data:<mime>;base64,<payload>"
                    data_url = evt.get("data", "")
                    if data_url.startswith("data:") and ";base64," in data_url:
                        header, b64 = data_url.split(",", 1)
                        mime = header[5:].split(";")[0] or "image/jpeg"
                        raw = base64.b64decode(b64)
                        # Analyze
                        resp = generate_image_response(
                            "Please analyze this image and provide relevant information.",
                            raw, st.session_state.model_name,
                            st.session_state.project_id, st.session_state.location,
                            st.session_state.creds, mime_type=mime
                        )
                        st.session_state.messages.append({"role": "assistant", "content": f"**Image Analysis:**\n\n{resp}"})
                        st.rerun()
        except Exception as e:
            st.error(f"Handler error: {e}")

if __name__ == "__main__":
    main()
