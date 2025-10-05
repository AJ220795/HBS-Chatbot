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
KB_DIR = APP_DIR / "kb"
EXTRACT_DIR = DATA_DIR / "kb_extracted"
INDEX_PATH = DATA_DIR / "faiss.index"
CORPUS_PATH = DATA_DIR / "corpus.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_MODELS = [
    "gemini-2.5-flash-lite",
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

def parse_report_data_from_ocr(ocr_text: str, filename: str) -> List[Dict]:
    """Parse OCR text to extract structured report data"""
    structured_data = []
    
    # Helper function to parse overdue report
    def parse_overdue_report(text: str) -> List[Dict]:
        data = []
        lines = text.split('\n')
        current_unit = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for stock number pattern (e.g., "12345")
            if re.match(r'^\d+$', line) and len(line) >= 3:
                if current_unit:
                    data.append(current_unit)
                current_unit = {'stock_number': line, 'type': 'overdue_unit'}
            
            # Look for customer info
            elif 'customer' in line.lower() or 'contract' in line.lower():
                current_unit['customer_info'] = line
            
            # Look for dates
            elif re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', line):
                current_unit['date'] = line
        
        if current_unit:
            data.append(current_unit)
        return data
    
    # Helper function to parse outbound report
    def parse_outbound_report(text: str) -> List[Dict]:
        data = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for equipment entries
            if re.search(r'\d+.*\d{4}', line):  # Pattern like "12345 BOB T650 SKIDSTEER 2023"
                parts = line.split()
                if len(parts) >= 4:
                    data.append({
                        'stock_number': parts[0],
                        'make': parts[1],
                        'model': parts[2],
                        'year': parts[-1],
                        'type': 'outbound_equipment'
                    })
        
        return data
    
    # Helper function to parse equipment list report
    def parse_equipment_list_report(text: str) -> List[Dict]:
        data = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for equipment entries with stock numbers
            if re.search(r'^\d+', line) and len(line.split()) >= 3:
                parts = line.split()
                data.append({
                    'stock_number': parts[0],
                    'equipment_info': ' '.join(parts[1:]),
                    'type': 'equipment_list'
                })
        
        return data
    
    # Determine report type and parse accordingly
    text_lower = ocr_text.lower()
    
    if 'overdue' in text_lower:
        structured_data.extend(parse_overdue_report(ocr_text))
    elif 'outbound' in text_lower:
        structured_data.extend(parse_outbound_report(ocr_text))
    elif 'equipment list' in text_lower or 'rental equipment' in text_lower:
        structured_data.extend(parse_equipment_list_report(ocr_text))
    
    return structured_data

def embed_texts(texts: List[str], project_id: str, location: str, credentials) -> np.ndarray:
    """Generate embeddings for texts using text-embedding-005"""
    vertexai_init(project=project_id, location=location, credentials=credentials)
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    
    embeddings = []
    for text in texts:
        try:
            emb = model.get_embeddings([text])[0].values
            embeddings.append(emb)
        except Exception as e:
            st.error(f"Error embedding text: {e}")
            # Use zero vector as fallback
            embeddings.append([0.0] * 768)
    
    return np.array(embeddings)

def build_faiss_index(corpus: List[Dict], project_id: str, location: str, credentials) -> tuple:
    """Build FAISS index from corpus"""
    texts = [item['text'] for item in corpus]
    embeddings = embed_texts(texts, project_id, location, credentials)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    return index, embeddings

def search_index(query: str, index, corpus: List[Dict], project_id: str, location: str, credentials, k: int = 2, min_similarity: float = 0.5) -> List[Dict]:
    """Search FAISS index with similarity threshold"""
    # Generate query embedding
    query_embedding = embed_texts([query], project_id, location, credentials)[0]
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding.astype('float32'), k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= min_similarity:  # Only include results above threshold
            chunk = corpus[idx].copy()
            chunk['similarity_score'] = float(score)
            results.append(chunk)
    
    return results

def load_index_and_corpus() -> tuple:
    """Load saved FAISS index and corpus"""
    try:
        if INDEX_PATH.exists() and CORPUS_PATH.exists():
            index = faiss.read_index(str(INDEX_PATH))
            with open(CORPUS_PATH, 'r') as f:
                corpus = json.load(f)
            return index, corpus
    except Exception as e:
        st.error(f"Error loading index: {e}")
    return None, None

def save_index_and_corpus(index, corpus: List[Dict]):
    """Save FAISS index and corpus"""
    try:
        faiss.write_index(index, str(INDEX_PATH))
        with open(CORPUS_PATH, 'w') as f:
            json.dump(corpus, f, indent=2)
    except Exception as e:
        st.error(f"Error saving index: {e}")

def process_kb_files() -> List[Dict]:
    """Process all KB files and create corpus"""
    corpus = []
    
    # Debug: List all files in KB directory
    kb_files = list(KB_DIR.glob("*"))
    st.write(f"DEBUG: Found {len(kb_files)} files in KB directory:")
    for f in kb_files:
        st.write(f"  - {f.name} ({f.stat().st_size} bytes)")
    
    for file_path in kb_files:
        if file_path.is_file():
            try:
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                
                # Extract text based on file type
                if file_path.suffix.lower() == '.docx':
                    text = extract_text_from_docx_bytes(file_bytes)
                elif file_path.suffix.lower() == '.pdf':
                    text = extract_text_from_pdf_bytes(file_bytes)
                elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    text = extract_text_from_image_bytes(file_bytes)
                else:
                    continue
                
                if text.strip():
                    # Parse structured data from OCR text
                    structured_data = parse_report_data_from_ocr(text, file_path.name)
                    
                    # Create chunks
                    chunks = chunk_text(text)
                    
                    # Add to corpus
                    for i, chunk in enumerate(chunks):
                        corpus.append({
                            'text': chunk,
                            'source': file_path.name,
                            'chunk_id': i,
                            'file_type': file_path.suffix.lower(),
                            'structured_data': structured_data
                        })
                    
                    st.write(f"✅ Processed {file_path.name}: {len(chunks)} chunks, {len(structured_data)} structured items")
                else:
                    st.write(f"⚠️ No text extracted from {file_path.name}")
                    
            except Exception as e:
                st.error(f"Error processing {file_path.name}: {e}")
    
    return corpus

def get_conversation_context(messages: List[Dict]) -> str:
    """Extract conversation context from recent messages"""
    if len(messages) < 2:
        return ""
    
    # Get last 3 exchanges (6 messages)
    recent_messages = messages[-6:]
    context_parts = []
    
    for msg in recent_messages:
        role = msg['role'].title()
        content = msg['content'][:200]  # Limit length
        context_parts.append(f"{role}: {content}")
    
    return " | ".join(context_parts)

def classify_user_intent(query: str, conversation_context: str, model_name: str, project_id: str, location: str, credentials) -> Dict:
    """Use LLM to classify user intent with improved error handling"""
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        
        # Simplified prompt for better reliability
        prompt = f"""Analyze this user query and conversation context to determine the user's intent.

User Query: "{query}"
Conversation Context: "{conversation_context}"

Classify the intent as one of these categories:
- "greeting": Hello, hi, hey, good morning, etc.
- "farewell": Goodbye, bye, see you later, etc.
- "clarification": I don't understand, can you explain, what does that mean, etc.
- "troubleshooting": It's not working, I tried that, I can't find it, etc.
- "alternative": Any other way, alternatives, other methods, etc.
- "new_question": A completely new question or topic

Respond with ONLY a JSON object in this exact format:
{{"intent": "category", "confidence": 0.8, "reasoning": "brief explanation"}}"""

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up the response text
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Try to parse JSON
        try:
            result = json.loads(response_text)
            # Validate the result
            if 'intent' in result and result['intent'] in ['greeting', 'farewell', 'clarification', 'troubleshooting', 'alternative', 'new_question']:
                return result
            else:
                return {"intent": "new_question", "confidence": 0.5, "reasoning": "Invalid intent category"}
        except json.JSONDecodeError:
            # Try to extract intent from text
            if 'clarification' in response_text.lower() or 'understand' in response_text.lower():
                return {"intent": "clarification", "confidence": 0.7, "reasoning": "Extracted from text"}
            elif 'troubleshooting' in response_text.lower() or 'working' in response_text.lower():
                return {"intent": "troubleshooting", "confidence": 0.7, "reasoning": "Extracted from text"}
            elif 'alternative' in response_text.lower() or 'other' in response_text.lower():
                return {"intent": "alternative", "confidence": 0.7, "reasoning": "Extracted from text"}
            else:
                return {"intent": "new_question", "confidence": 0.5, "reasoning": "Failed to parse, defaulting to new_question"}
                
    except Exception as e:
        st.error(f"Error in intent classification: {e}")
        return {"intent": "new_question", "confidence": 0.5, "reasoning": f"Error: {str(e)}"}

def get_conversational_response(query: str) -> str:
    """Handle simple conversational queries"""
    query_lower = query.lower().strip()
    
    # Greetings
    if any(word in query_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return "Hi! How can I help you?"
    
    # Farewells
    if any(word in query_lower for word in ['bye', 'goodbye', 'see you', 'thanks', 'thank you']):
        return "You're welcome! Feel free to ask if you need any more help with HBS systems."
    
    # Casual responses
    if any(word in query_lower for word in ['how are you', 'how\'s it going', 'what\'s up']):
        return "I'm doing well, thank you! I'm here to help you with HBS systems and reports. What can I assist you with today?"
    
    return None

def generate_response(query: str, context_chunks: List[Dict], model_name: str, project_id: str, location: str, credentials, intent: str = "new_question") -> str:
    """Generate response using Gemini with context and intent awareness"""
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        
        # Prepare context
        context_text = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Extract structured data from context
        structured_data = []
        for chunk in context_chunks:
            if chunk.get('structured_data'):
                structured_data.extend(chunk['structured_data'])
        
        structured_text = ""
        if structured_data:
            structured_text = f"\n\nStructured Data Found:\n{json.dumps(structured_data, indent=2)}"
        
        # Intent-aware system prompt
        intent_guidance = ""
        if intent == "clarification":
            intent_guidance = "The user is asking for clarification. Provide a clear, detailed explanation of the previous topic discussed."
        elif intent == "troubleshooting":
            intent_guidance = "The user is having trouble with something. Provide specific troubleshooting steps and ask for more details about what's not working."
        elif intent == "alternative":
            intent_guidance = "The user is asking for alternative methods. Acknowledge what you've shared and explain if you don't have information about other approaches."
        
        system_prompt = f"""You are a helpful HBS (Heavy Business Systems) support assistant. You help users with HBS software, reports, and procedures.

Context from Knowledge Base:
{context_text}{structured_text}

Instructions:
1. Answer questions based on the provided context
2. Be specific and provide step-by-step instructions when possible
3. If the context doesn't contain enough information, say so clearly
4. Use the structured data to provide specific examples (stock numbers, dates, etc.)
5. Be helpful and professional
6. {intent_guidance}

User Question: {query}

Provide a helpful response based on the context above."""

        response = model.generate_content(system_prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

def generate_image_response(query: str, image_bytes: bytes, model_name: str, project_id: str, location: str, credentials) -> str:
    """Generate response for image-based queries"""
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        
        # Convert image bytes to PIL Image
        image = PILImage.open(io.BytesIO(image_bytes))
        
        # Create the prompt
        prompt = f"""Analyze this image and answer the user's question: {query}

If this appears to be an HBS report or document, provide specific details about:
- Report type and purpose
- Key data points (stock numbers, dates, totals, etc.)
- Any notable patterns or issues
- How this relates to HBS system functionality

Be specific and reference actual data from the image when possible."""

        response = model.generate_content([prompt, Image(image)])
        return response.text
        
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return "I apologize, but I encountered an error while analyzing the image. Please try again."

def initialize_app():
    """Initialize the app and load/build index - FIXED VERSION"""
    try:
        # Load credentials
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["vertex_ai_credentials"]
        )
        project_id = st.secrets["project_id"]
        location = st.secrets.get("location", DEFAULT_LOCATION)
        
        # Try to load existing index
        index, corpus = load_index_and_corpus()
        
        if index is None or corpus is None:
            st.info("Building knowledge base index...")
            corpus = process_kb_files()
            
            if corpus:
                index, _ = build_faiss_index(corpus, project_id, location, credentials)
                save_index_and_corpus(index, corpus)
                st.success(f"✅ Knowledge base built with {len(corpus)} chunks from {len(set(item['source'] for item in corpus))} files")
            else:
                st.error("❌ No files found in knowledge base directory")
                return None, None, None, None
        
        return index, corpus, project_id, location, credentials
        
    except Exception as e:
        st.error(f"Error initializing app: {e}")
        return None, None, None, None

def main():
    st.set_page_config(
        page_title="HBS Help Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb_loaded" not in st.session_state:
        st.session_state.kb_loaded = False
    
    # Initialize app - FIXED: Removed @st.cache_resource decorator
    index, corpus, project_id, location, credentials = initialize_app()
    
    if index is None:
        st.error("Failed to initialize app. Please check your configuration.")
        return
    
    # Set KB loaded status
    st.session_state.kb_loaded = True
    
    # Sidebar
    with st.sidebar:
        st.title("HBS Help Chatbot")
        
        # Status
        if st.session_state.kb_loaded:
            st.success("✅ Knowledge Base Loaded")
            st.info(f"📚 {len(corpus)} chunks from {len(set(item['source'] for item in corpus))} files")
        else:
            st.error("❌ Knowledge Base Not Loaded")
        
        # Model selection
        model_name = st.selectbox(
            "Select Model:",
            CANDIDATE_MODELS,
            index=0,  # Default to gemini-2.5-flash-lite
            key="model_select"
        )
        
        # Rebuild index button
        if st.button("🔄 Rebuild Index", key="rebuild_btn"):
            st.session_state.kb_loaded = False
            # Clear any cached data
            if 'index' in st.session_state:
                del st.session_state['index']
            if 'corpus' in st.session_state:
                del st.session_state['corpus']
            st.rerun()
    
    # Main chat interface
    st.title("HBS Help Chatbot")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message.get("sources"):
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**{source['source']}** (similarity: {source['similarity_score']:.3f})")
                        st.write(source['text'][:200] + "...")
    
    # Initial greeting
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("Hi! How can I help you?")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hi! How can I help you?",
            "sources": [],
            "timestamp": 0
        })
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about HBS systems..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": len(st.session_state.messages)
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if this is a conversational query first
        conversational_response = get_conversational_response(prompt)
        
        # Get conversation context
        conversation_context = get_conversation_context(st.session_state.messages)
        
        # Debug output
        st.write(f"🔍 DEBUG: Conversation context: '{conversation_context}'")
        st.write(f"🔍 DEBUG: Messages count: {len(st.session_state.messages)}")
        st.write(f"🔍 DEBUG: Last few messages: {st.session_state.messages[-3:] if len(st.session_state.messages) >= 3 else st.session_state.messages}")
        
        # Classify user intent
        intent_result = classify_user_intent(prompt, conversation_context, model_name, project_id, location, credentials)
        st.write(f"🔍 DEBUG: Intent detected: {intent_result}")
        
        # Handle different intents
        if conversational_response:
            # Simple conversational response
            with st.chat_message("assistant"):
                st.markdown(conversational_response)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": conversational_response,
                "sources": [],
                "timestamp": len(st.session_state.messages)
            })
        else:
            # Search knowledge base
            context_chunks = search_index(prompt, index, corpus, project_id, location, credentials)
            
            if context_chunks:
                # Generate response with context and intent
                intent = intent_result.get('intent', 'new_question')
                response = generate_response(prompt, context_chunks, model_name, project_id, location, credentials, intent)
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Display sources
                if context_chunks:
                    with st.expander("Sources"):
                        for i, chunk in enumerate(context_chunks[:2]):  # Limit to 2 sources
                            # Create clickable source link
                            source_name = chunk['source']
                            similarity = chunk['similarity_score']
                            content_preview = chunk['text'][:200] + "..."
                            
                            # Try to create a clickable link to the source file
                            try:
                                # Check if source file exists in KB directory
                                source_path = KB_DIR / source_name
                                if source_path.exists():
                                    # Create a download link
                                    with open(source_path, 'rb') as f:
                                        file_data = f.read()
                                    
                                    st.download_button(
                                        label=f"📄 {source_name} (similarity: {similarity:.3f})",
                                        data=file_data,
                                        file_name=source_name,
                                        mime="application/octet-stream",
                                        key=f"download_{i}_{len(st.session_state.messages)}"
                                    )
                                else:
                                    st.write(f"📄 {source_name} (similarity: {similarity:.3f})")
                            except Exception as e:
                                st.write(f"📄 {source_name} (similarity: {similarity:.3f})")
                            
                            st.write(content_preview)
                            st.write("---")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": context_chunks,
                    "timestamp": len(st.session_state.messages)
                })
            else:
                # No relevant context found - use intent-aware responses
                intent = intent_result.get('intent', 'new_question')
                
                if intent == 'clarification':
                    response = "I'd be happy to clarify! Based on our previous conversation, could you let me know which specific part you'd like me to explain better?"
                elif intent == 'troubleshooting':
                    response = "I understand you're having trouble. Could you provide more details about what's not working? For example, are you seeing any error messages or is a specific step not working as expected?"
                elif intent == 'alternative':
                    response = "I don't have information about alternative methods for that topic in my knowledge base. Based on what I've shared, the main approach is what I described earlier. If you need help with a different approach, please let me know what specifically you're looking for."
                else:
                    response = "I don't have information about that topic in my knowledge base. Could you please rephrase your question or ask about HBS reports, procedures, or system features?"
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": [],
                    "timestamp": len(st.session_state.messages)
                })
    
    # Image upload handling
    uploaded_file = st.file_uploader("Upload an image for analysis", type=['png', 'jpg', 'jpeg'], key="image_upload")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = PILImage.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add image analysis button
        if st.button("Analyze Image", key="analyze_btn"):
            # Generate response for image
            image_bytes = uploaded_file.getvalue()
            response = generate_image_response("Analyze this image and provide details about any HBS reports or data shown", image_bytes, model_name, project_id, location, credentials)
            
            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": [],
                "timestamp": len(st.session_state.messages)
            })
            
            # Clear the uploaded file
            st.rerun()

if __name__ == "__main__":
    main()
