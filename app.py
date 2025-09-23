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
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
]

DEFAULT_LOCATION = "us-central1"

# ---- Utilities ----
def split_into_sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def chunk_text(text: str, max_tokens: int = 800, overlap_sentences: int = 2) -> List[str]:
    """Improved chunking with better overlap and context preservation"""
    sents = split_into_sentences(text)
    chunks, buf, token_est = [], [], 0
    
    for s in sents:
        s_tokens = max(1, len(s) // 4)
        if token_est + s_tokens > max_tokens and buf:
            chunks.append(" ".join(buf))
            # Better overlap strategy
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

def embed_texts(texts: List[str], project_id: str, location: str, credentials) -> np.ndarray:
    """Generate embeddings for texts"""
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        embeddings = model.get_embeddings(texts)
        return np.array([e.values for e in embeddings]).astype(np.float32)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return np.array([])

def build_faiss_index(corpus: List[Dict], project_id: str, location: str, credentials) -> tuple:
    """Build FAISS index from corpus"""
    if not corpus:
        return None, []
    
    texts = [item["text"] for item in corpus]
    embeddings = embed_texts(texts, project_id, location, credentials)
    
    if embeddings.size == 0:
        return None, []
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    return index, corpus

def search_index(query: str, index, corpus: List[Dict], project_id: str, location: str, credentials, k: int = 5) -> List[Dict]:
    """Search FAISS index for relevant chunks"""
    if index is None or not corpus:
        return []
    
    try:
        # Generate query embedding
        query_embeddings = embed_texts([query], project_id, location, credentials)
        if query_embeddings.size == 0:
            return []
        
        # Normalize query embedding
        faiss.normalize_L2(query_embeddings)
        
        # Search
        scores, indices = index.search(query_embeddings, min(k, len(corpus)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(corpus):
                results.append({
                    **corpus[idx],
                    "similarity_score": float(score)
                })
        
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def load_index_and_corpus() -> tuple:
    """Load existing FAISS index and corpus"""
    try:
        if INDEX_PATH.exists() and CORPUS_PATH.exists():
            index = faiss.read_index(str(INDEX_PATH))
            with open(CORPUS_PATH, 'r') as f:
                corpus = json.load(f)
            return index, corpus
    except Exception as e:
        st.error(f"Error loading index: {e}")
    return None, []

def save_index_and_corpus(index, corpus: List[Dict]):
    """Save FAISS index and corpus"""
    try:
        if index is not None:
            faiss.write_index(index, str(INDEX_PATH))
        with open(CORPUS_PATH, 'w') as f:
            json.dump(corpus, f, indent=2)
    except Exception as e:
        st.error(f"Error saving index: {e}")

def process_kb_files() -> List[Dict]:
    """Process all KB files and create corpus"""
    corpus = []
    
    if not KB_DIR.exists():
        return corpus
    
    for file_path in KB_DIR.iterdir():
        if file_path.is_file():
            try:
                if file_path.suffix.lower() == '.docx':
                    text = extract_text_from_docx_bytes(file_path.read_bytes())
                elif file_path.suffix.lower() == '.pdf':
                    text = extract_text_from_pdf_bytes(file_path.read_bytes())
                elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
                    text = extract_text_from_image_bytes(file_path.read_bytes())
                else:
                    continue
                
                if text.strip():
                    chunks = chunk_text(text)
                    for i, chunk in enumerate(chunks):
                        corpus.append({
                            "text": chunk,
                            "source": file_path.name,
                            "chunk_id": i,
                            "file_type": file_path.suffix.lower()
                        })
            except Exception as e:
                st.error(f"Error processing {file_path.name}: {e}")
    
    return corpus

def get_conversational_response(query: str) -> str:
    """Handle simple conversational queries"""
    query_lower = query.lower().strip()
    
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    farewells = ["bye", "goodbye", "see you", "farewell", "thanks", "thank you"]
    casual = ["what's up", "how are you", "how's it going", "what's new"]
    
    if any(greeting in query_lower for greeting in greetings):
        return "Hello! I'm your HBS Help Chatbot. I can help you with questions about HBS systems, features, and procedures. What would you like to know?"
    
    if any(farewell in query_lower for farewell in farewells):
        return "Goodbye! Feel free to come back anytime if you have more questions about HBS systems."
    
    if any(casual_phrase in query_lower for casual_phrase in casual):
        return "I'm doing well, thank you! I'm here to help you with any questions about HBS systems, reports, or procedures. What can I assist you with today?"
    
    return None

def generate_response(query: str, context_chunks: List[Dict], model_name: str, project_id: str, location: str, credentials) -> str:
    """Generate response using Gemini with improved prompting"""
    
    # Check for conversational responses first
    conversational = get_conversational_response(query)
    if conversational:
        return conversational
    
    if not context_chunks:
        return "I don't have specific information about that topic in my knowledge base. Could you please rephrase your question or ask about HBS reports, procedures, or system features?"
    
    # Build context from retrieved chunks
    context_text = "\n\n".join([
        f"Source: {chunk['source']}\nContent: {chunk['text']}"
        for chunk in context_chunks
    ])
    
    # Improved system prompt
    system_prompt = f"""You are an expert HBS (Help Business System) assistant. You have access to detailed documentation about HBS systems, reports, and procedures.

CONTEXT FROM KNOWLEDGE BASE:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the user's question using ONLY the information provided in the context above
2. Be specific and detailed in your response
3. If the context contains step-by-step procedures, provide them clearly
4. If the context mentions specific options, fields, or settings, list them exactly
5. If you need to make assumptions, state them clearly
6. If the context doesn't contain enough information to fully answer the question, say so and suggest what additional information might be needed
7. Always be helpful and professional

RESPONSE:"""

    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        
        response = model.generate_content(
            system_prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=2048,
                top_p=0.8,
                top_k=40
            )
        )
        
        return response.text if response.text else "I couldn't generate a response. Please try rephrasing your question."
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_image_response(query: str, image_bytes: bytes, model_name: str, project_id: str, location: str, credentials) -> str:
    """Generate response for image-based queries"""
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        
        # Create image part
        image_part = Part.from_data(image_bytes, mime_type="image/jpeg")
        
        prompt = f"""Analyze this image and answer the user's question: {query}

If this appears to be a screenshot or document related to HBS systems, provide detailed analysis. If it's not related to HBS, politely explain that you specialize in HBS system assistance."""
        
        response = model.generate_content([prompt, image_part])
        return response.text if response.text else "I couldn't analyze the image. Please try again."
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# ---- Streamlit App ----
def main():
    st.set_page_config(
        page_title="HBS Help Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
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
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    # Initialize credentials
    try:
        sa_info = json.loads(st.secrets["google"]["credentials_json"])
        st.session_state.creds = service_account.Credentials.from_service_account_info(sa_info)
        st.session_state.project_id = st.secrets["google"]["project"]
        st.session_state.location = st.secrets["google"].get("location", DEFAULT_LOCATION)
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        st.stop()

    # Initialize app
    @st.cache_resource
    def initialize_app():
        """Initialize the app - load index or build from KB files"""
        # Try to load existing index first
        index, corpus = load_index_and_corpus()
        if index is not None and corpus:
            return index, corpus, True
        
        # Build index from KB files
        corpus = process_kb_files()
        if not corpus:
            return None, [], False
        
        index, corpus = build_faiss_index(corpus, st.session_state.project_id, st.session_state.location, st.session_state.creds)
        if index is not None:
            save_index_and_corpus(index, corpus)
            return index, corpus, True
        
        return None, [], False

    # Initialize
    if not st.session_state.kb_loaded:
        with st.spinner("Loading knowledge base..."):
            index, corpus, loaded = initialize_app()
            st.session_state.index = index
            st.session_state.corpus = corpus
            st.session_state.kb_loaded = loaded

    # Sidebar
    with st.sidebar:
        st.header("HBS Help Chatbot")
        
        # Status
        if st.session_state.kb_loaded:
            st.success(f"✅ Knowledge base loaded ({len(st.session_state.corpus)} chunks)")
        else:
            st.error("❌ Knowledge base not loaded")
        
        # Model selection
        st.subheader("Model Settings")
        st.session_state.model_name = st.selectbox(
            "Select Model",
            CANDIDATE_MODELS,
            index=0,
            key="model_select"
        )
        
        # Rebuild index button
        if st.button("🔄 Rebuild Index", key="rebuild_btn"):
            with st.spinner("Rebuilding index..."):
                corpus = process_kb_files()
                if corpus:
                    index, corpus = build_faiss_index(corpus, st.session_state.project_id, st.session_state.location, st.session_state.creds)
                    if index is not None:
                        save_index_and_corpus(index, corpus)
                        st.session_state.index = index
                        st.session_state.corpus = corpus
                        st.session_state.kb_loaded = True
                        st.success("Index rebuilt successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to build index")
                else:
                    st.error("No KB files found")

    # Main chat interface
    st.title("HBS Help Chatbot")
    
    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"**{source['source']}** (similarity: {source['similarity_score']:.3f})")
                        st.write(source['text'][:200] + "...")

    # Chat input
    if prompt := st.chat_input("Ask me anything about HBS systems..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Search for relevant context
                context_chunks = search_index(
                    prompt, 
                    st.session_state.index, 
                    st.session_state.corpus,
                    st.session_state.project_id,
                    st.session_state.location,
                    st.session_state.creds,
                    k=5
                )
                
                # Generate response
                response = generate_response(
                    prompt,
                    context_chunks,
                    st.session_state.model_name,
                    st.session_state.project_id,
                    st.session_state.location,
                    st.session_state.creds
                )
                
                st.markdown(response)
                
                # Show sources if available
                if context_chunks:
                    with st.expander("Sources"):
                        for chunk in context_chunks:
                            st.write(f"**{chunk['source']}** (similarity: {chunk['similarity_score']:.3f})")
                            st.write(chunk['text'][:200] + "...")
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": context_chunks
        })

    # Image upload (simplified)
    if st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Ask about this image", key="ask_image_btn"):
            with st.spinner("Analyzing image..."):
                image_bytes = st.session_state.uploaded_image.getvalue()
                response = generate_image_response(
                    "What do you see in this image?",
                    image_bytes,
                    st.session_state.model_name,
                    st.session_state.project_id,
                    st.session_state.location,
                    st.session_state.creds
                )
                st.write(response)
                st.session_state.uploaded_image = None
                st.rerun()

if __name__ == "__main__":
    main()
