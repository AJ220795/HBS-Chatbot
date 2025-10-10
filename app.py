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

# LangChain imports 
try:
    from langchain_google_vertexai import VertexAI
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import ConversationChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("LangChain not available. Using fallback conversation handling.")

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
    "gemini-2.5-pro"
]

DEFAULT_LOCATION = "us-central1"

# Token limits for different contexts
MAX_CONTEXT_TOKENS = 150000  # Leave room for response and conversation
MAX_CHUNKS_INITIAL = 50  # Retrieve many chunks initially
MAX_CHUNKS_FINAL = 10    # Send top 10 to model after re-ranking

# ---- Token Counting Utilities ----
def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
    return len(text) // 4

def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit"""
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text
    
    # Truncate to approximate character limit
    char_limit = max_tokens * 4
    return text[:char_limit] + "...[truncated]"

# ---- Utilities ----
def split_into_sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def chunk_text(text: str, max_tokens: int = 500, overlap_sentences: int = 2) -> List[str]:
    """Improved chunking with smaller chunks for better retrieval"""
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
    
    # Clean up OCR text
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    
    # Look for report patterns
    if "overdue" in filename.lower() or "overdue" in ocr_text.lower():
        structured_data.extend(parse_overdue_report(ocr_text, filename))
    elif "outbound" in filename.lower() or "outbound" in ocr_text.lower():
        structured_data.extend(parse_outbound_report(ocr_text, filename))
    elif "equipment list" in filename.lower() or "equipment list" in ocr_text.lower():
        structured_data.extend(parse_equipment_list_report(ocr_text, filename))
    
    return structured_data

def parse_overdue_report(ocr_text: str, filename: str) -> List[Dict]:
    """Parse Overdue Equipment Report data"""
    data = []
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    
    for i, line in enumerate(lines):
        if re.match(r'^[A-Z\s]+$', line) and len(line) > 3:
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                contract_match = re.search(r'C\d+R', next_line)
                if contract_match:
                    customer_name = line
                    contract = contract_match.group()
                    
                    phone = ""
                    stock = ""
                    make = ""
                    model = ""
                    equipment_type = ""
                    year = ""
                    serial = ""
                    date_out = ""
                    expected = ""
                    days_over = ""
                    
                    for j in range(i, min(i + 5, len(lines))):
                        current_line = lines[j]
                        
                        phone_match = re.search(r'\(\d{3}\)\s*\d{3}-\d{4}', current_line)
                        if phone_match:
                            phone = phone_match.group()
                        
                        stock_match = re.search(r'\b\d{5}\b', current_line)
                        if stock_match:
                            stock = stock_match.group()
                        
                        make_match = re.search(r'\b(BOB|KUB|JD|BOM)\b', current_line)
                        if make_match:
                            make = make_match.group()
                        
                        model_match = re.search(r'\b(T650|E32|E42|U55-4R3AP|U35-4R3A|E26|KX080R3AT3|KX121R3TA|KX121RRATS|211D-50|690B|442)\b', current_line)
                        if model_match:
                            model = model_match.group()
                        
                        type_match = re.search(r'\b(SKIDSTEER|EXCAVATOR|ROLLER)\b', current_line)
                        if type_match:
                            equipment_type = type_match.group()
                        
                        year_match = re.search(r'\b(2013|2014|2015|2016|1979|2006|2008|2012)\b', current_line)
                        if year_match:
                            year = year_match.group()
                        
                        serial_match = re.search(r'\b[A-Z0-9]{6,}\b', current_line)
                        if serial_match and len(serial_match.group()) > 6:
                            serial = serial_match.group()
                        
                        date_match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', current_line)
                        if date_match:
                            if not date_out:
                                date_out = date_match.group()
                            else:
                                expected = date_match.group()
                        
                        days_match = re.search(r'\b\d{1,4}\b', current_line)
                        if days_match and days_match.group().isdigit():
                            days_over = days_match.group()
                    
                    if customer_name and contract:
                        data.append({
                            "customer_name": customer_name,
                            "contract": contract,
                            "phone": phone,
                            "stock_number": stock,
                            "make": make,
                            "model": model,
                            "equipment_type": equipment_type,
                            "year": year,
                            "serial": serial,
                            "date_out": date_out,
                            "expected_due": expected,
                            "days_overdue": days_over,
                            "source": filename,
                            "report_type": "Overdue Equipment Report"
                        })
    
    return data

def parse_outbound_report(ocr_text: str, filename: str) -> List[Dict]:
    """Parse Outbound Report data"""
    data = []
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    
    for i, line in enumerate(lines):
        if re.match(r'^[A-Z\s]+$', line) and len(line) > 3:
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                contract_match = re.search(r'C\d+R', next_line)
                if contract_match:
                    customer_name = line
                    contract = contract_match.group()
                    
                    phone = ""
                    stock = ""
                    make = ""
                    model = ""
                    equipment_type = ""
                    year = ""
                    serial = ""
                    date_time_out = ""
                    
                    for j in range(i, min(i + 5, len(lines))):
                        current_line = lines[j]
                        
                        phone_match = re.search(r'\(\d{3}\)\s*\d{3}-\d{4}', current_line)
                        if phone_match:
                            phone = phone_match.group()
                        
                        stock_match = re.search(r'\b\d{5}\b', current_line)
                        if stock_match:
                            stock = stock_match.group()
                        
                        make_match = re.search(r'\b(BOB|KUB|JD|BOM)\b', current_line)
                        if make_match:
                            make = make_match.group()
                        
                        model_match = re.search(r'\b(T650|E32|E42|U55-4R3AP|U35-4R3A|E26|KX080R3AT3|KX121R3TA|KX121RRATS|211D-50|690B|442)\b', current_line)
                        if model_match:
                            model = model_match.group()
                        
                        type_match = re.search(r'\b(SKIDSTEER|EXCAVATOR|ROLLER)\b', current_line)
                        if type_match:
                            equipment_type = type_match.group()
                        
                        year_match = re.search(r'\b(2013|2014|2015|2016|1979|2006|2008|2012)\b', current_line)
                        if year_match:
                            year = year_match.group()
                        
                        serial_match = re.search(r'\b[A-Z0-9]{6,}\b', current_line)
                        if serial_match and len(serial_match.group()) > 6:
                            serial = serial_match.group()
                        
                        datetime_match = re.search(r'\b\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}\s+[AP]M\b', current_line)
                        if datetime_match:
                            date_time_out = datetime_match.group()
                    
                    if customer_name and contract:
                        data.append({
                            "customer_name": customer_name,
                            "contract": contract,
                            "phone": phone,
                            "stock_number": stock,
                            "make": make,
                            "model": model,
                            "equipment_type": equipment_type,
                            "year": year,
                            "serial": serial,
                            "date_time_out": date_time_out,
                            "source": filename,
                            "report_type": "Rental Outbound Report"
                        })
    
    return data

def parse_equipment_list_report(ocr_text: str, filename: str) -> List[Dict]:
    """Parse Equipment List Report data"""
    data = []
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    
    for i, line in enumerate(lines):
        stock_match = re.search(r'\b\d{5}\b', line)
        if stock_match:
            stock = stock_match.group()
            
            make = ""
            model = ""
            equipment_type = ""
            year = ""
            serial = ""
            location = ""
            meter = ""
            
            make_match = re.search(r'\b(BOB|KUB|JD|BOM)\b', line)
            if make_match:
                make = make_match.group()
            
            model_match = re.search(r'\b(T650|E32|E42|U55-4R3AP|U35-4R3A|E26|KX080R3AT3|KX121R3TA|KX121RRATS|211D-50|690B|442)\b', line)
            if model_match:
                model = model_match.group()
            
            type_match = re.search(r'\b(SKIDSTEER|EXCAVATOR|ROLLER)\b', line)
            if type_match:
                equipment_type = type_match.group()
            
            year_match = re.search(r'\b(2013|2014|2015|2016|1979|2006|2008|2012)\b', line)
            if year_match:
                year = year_match.group()
            
            serial_match = re.search(r'\b[A-Z0-9]{6,}\b', line)
            if serial_match and len(serial_match.group()) > 6:
                serial = serial_match.group()
            
            meter_match = re.search(r'\b\d+\b', line)
            if meter_match:
                meter = meter_match.group()
            
            data.append({
                "stock_number": stock,
                "make": make,
                "model": model,
                "equipment_type": equipment_type,
                "year": year,
                "serial": serial,
                "location": location,
                "meter": meter,
                "source": filename,
                "report_type": "Rental Equipment List"
            })
    
    return data

def embed_texts(texts: List[str], project_id: str, location: str, credentials, batch_size: int = 250) -> np.ndarray:
    """Generate embeddings for texts in batches to handle large corpora"""
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            st.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = model.get_embeddings(batch)
            all_embeddings.extend([e.values for e in batch_embeddings])
        
        return np.array(all_embeddings).astype(np.float32)
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

def expand_query(query: str) -> str:
    """Expand query with related terms for better search"""
    query_lower = query.lower()
    
    if "overdue" in query_lower:
        return f"{query} overdue equipment report rental"
    elif "outbound" in query_lower:
        return f"{query} outbound report rental equipment"
    elif "equipment" in query_lower:
        return f"{query} equipment list rental"
    elif "customer" in query_lower:
        return f"{query} customer contract phone"
    elif "stock" in query_lower:
        return f"{query} stock number equipment"
    elif "serial" in query_lower:
        return f"{query} serial number equipment"
    else:
        return query

def rerank_chunks(query: str, chunks: List[Dict], model_name: str, project_id: str, location: str, credentials, top_k: int = 10) -> List[Dict]:
    """Use LLM to re-rank chunks by relevance to query"""
    if len(chunks) <= top_k:
        return chunks
    
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        
        # Create a prompt to score relevance
        chunks_text = ""
        for i, chunk in enumerate(chunks[:30]):  # Only re-rank top 30 to save time
            chunks_text += f"\n[{i}] {chunk['text'][:200]}..."
        
        rerank_prompt = f"""Score each chunk's relevance to the query on a scale of 0-10.

QUERY: {query}

CHUNKS:
{chunks_text}

Return ONLY a JSON array of scores in order, like: [8, 3, 9, 2, ...]"""

        response = model.generate_content(
            rerank_prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=500
            )
        )
        
        if response.text:
            try:
                scores = json.loads(response.text.strip())
                if isinstance(scores, list) and len(scores) == len(chunks[:30]):
                    # Add scores to chunks
                    for i, score in enumerate(scores):
                        chunks[i]['rerank_score'] = float(score)
                    
                    # Sort by rerank score
                    sorted_chunks = sorted(chunks[:30], key=lambda x: x.get('rerank_score', 0), reverse=True)
                    return sorted_chunks[:top_k]
            except:
                pass
        
        # Fallback: return top k by similarity
        return chunks[:top_k]
    
    except Exception as e:
        # Fallback: return top k by similarity
        return chunks[:top_k]

def build_optimized_context(chunks: List[Dict], max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """Build optimized context from chunks with token limiting"""
    if not chunks:
        return "No relevant information found in knowledge base."
    
    context_parts = []
    current_tokens = 0
    
    for chunk in chunks:
        # Format chunk
        chunk_text = f"Source: {chunk['source']}\nContent: {chunk['text']}\n"
        chunk_tokens = estimate_tokens(chunk_text)
        
        # Check if adding this chunk would exceed limit
        if current_tokens + chunk_tokens > max_tokens:
            # Try to add a truncated version
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 100:  # Only add if we have reasonable space
                truncated = truncate_to_token_limit(chunk['text'], remaining_tokens - 50)
                context_parts.append(f"Source: {chunk['source']}\nContent: {truncated}\n")
            break
        
        context_parts.append(chunk_text)
        current_tokens += chunk_tokens
    
    return "\n".join(context_parts)

def search_index(query: str, index, corpus: List[Dict], project_id: str, location: str, credentials, model_name: str, k: int = MAX_CHUNKS_FINAL, min_similarity: float = 0.3) -> List[Dict]:
    """Search FAISS index with multi-stage retrieval and re-ranking"""
    if index is None or not corpus:
        return []
    
    try:
        # Stage 1: Expand query for better search
        expanded_query = expand_query(query)
        
        # Stage 2: Generate query embedding
        query_embeddings = embed_texts([expanded_query], project_id, location, credentials, batch_size=1)
        if query_embeddings.size == 0:
            return []
        
        # Normalize query embedding
        faiss.normalize_L2(query_embeddings)
        
        # Stage 3: Retrieve many candidates
        initial_k = min(MAX_CHUNKS_INITIAL, len(corpus))
        scores, indices = index.search(query_embeddings, initial_k)
        
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(corpus) and score >= min_similarity:
                candidates.append({
                    **corpus[idx],
                    "similarity_score": float(score)
                })
        
        if not candidates:
            return []
        
        # Stage 4: Re-rank using LLM for better relevance
        reranked = rerank_chunks(query, candidates, model_name, project_id, location, credentials, top_k=k)
        
        return reranked
    
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
    """Process all KB files and create corpus with image data extraction"""
    corpus = []
    
    if not KB_DIR.exists():
        st.error(f"KB_DIR does not exist: {KB_DIR}")
        return corpus
    
    files = list(KB_DIR.iterdir())
    st.info(f"Found {len(files)} files in KB directory")
    
    processed_files = 0
    for file_path in files:
        if file_path.is_file():
            try:
                if file_path.suffix.lower() == '.docx':
                    text = extract_text_from_docx_bytes(file_path.read_bytes())
                    if text.strip():
                        chunks = chunk_text(text)
                        for i, chunk in enumerate(chunks):
                            corpus.append({
                                "text": chunk,
                                "source": file_path.name,
                                "chunk_id": i,
                                "file_type": file_path.suffix.lower()
                            })
                
                elif file_path.suffix.lower() == '.pdf':
                    text = extract_text_from_pdf_bytes(file_path.read_bytes())
                    if text.strip():
                        chunks = chunk_text(text)
                        for i, chunk in enumerate(chunks):
                            corpus.append({
                                "text": chunk,
                                "source": file_path.name,
                                "chunk_id": i,
                                "file_type": file_path.suffix.lower()
                            })
                
                elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
                    file_size = file_path.stat().st_size
                    if file_size == 0:
                        continue
                    
                    try:
                        ocr_text = extract_text_from_image_bytes(file_path.read_bytes())
                        
                        if ocr_text.strip():
                            chunks = chunk_text(ocr_text)
                            for i, chunk in enumerate(chunks):
                                corpus.append({
                                    "text": chunk,
                                    "source": file_path.name,
                                    "chunk_id": i,
                                    "file_type": file_path.suffix.lower(),
                                    "content_type": "ocr_text"
                                })
                            
                            try:
                                structured_data = parse_report_data_from_ocr(ocr_text, file_path.name)
                                
                                for data_item in structured_data:
                                    searchable_text = f"Report: {data_item.get('report_type', 'Unknown')} "
                                    if 'customer_name' in data_item:
                                        searchable_text += f"Customer: {data_item['customer_name']} "
                                    if 'contract' in data_item:
                                        searchable_text += f"Contract: {data_item['contract']} "
                                    if 'stock_number' in data_item:
                                        searchable_text += f"Stock: {data_item['stock_number']} "
                                    if 'make' in data_item:
                                        searchable_text += f"Make: {data_item['make']} "
                                    if 'model' in data_item:
                                        searchable_text += f"Model: {data_item['model']} "
                                    if 'equipment_type' in data_item:
                                        searchable_text += f"Type: {data_item['equipment_type']} "
                                    if 'year' in data_item:
                                        searchable_text += f"Year: {data_item['year']} "
                                    if 'serial' in data_item:
                                        searchable_text += f"Serial: {data_item['serial']} "
                                    if 'days_overdue' in data_item:
                                        searchable_text += f"Days Overdue: {data_item['days_overdue']} "
                                    if 'date_out' in data_item:
                                        searchable_text += f"Date Out: {data_item['date_out']} "
                                    if 'expected_due' in data_item:
                                        searchable_text += f"Expected Due: {data_item['expected_due']} "
                                    if 'date_time_out' in data_item:
                                        searchable_text += f"Date/Time Out: {data_item['date_time_out']} "
                                    if 'phone' in data_item:
                                        searchable_text += f"Phone: {data_item['phone']} "
                                    if 'location' in data_item:
                                        searchable_text += f"Location: {data_item['location']} "
                                    if 'meter' in data_item:
                                        searchable_text += f"Meter: {data_item['meter']} "
                                    
                                    corpus.append({
                                        "text": searchable_text,
                                        "source": file_path.name,
                                        "chunk_id": len(corpus),
                                        "file_type": file_path.suffix.lower(),
                                        "content_type": "structured_data",
                                        "structured_data": data_item
                                    })
                                
                            except Exception as e:
                                pass
                                
                    except Exception as e:
                        pass
                
            except Exception as e:
                st.error(f"Error processing {file_path.name}: {e}")
        
        processed_files += 1
    
    st.success(f"Processed {processed_files} files, created {len(corpus)} chunks")
    return corpus

def get_conversation_context(messages: List[Dict], max_tokens: int = 2000) -> str:
    """Extract conversation context from recent messages with token limiting"""
    if not messages or len(messages) < 2:
        return ""
    
    recent_messages = messages[-6:]
    
    context_parts = []
    current_tokens = 0
    
    for msg in reversed(recent_messages):
        role = msg["role"]
        content = msg["content"]
        msg_text = f"{role.capitalize()}: {content}"
        msg_tokens = estimate_tokens(msg_text)
        
        if current_tokens + msg_tokens > max_tokens:
            break
        
        context_parts.insert(0, msg_text)
        current_tokens += msg_tokens
    
    return "\n".join(context_parts)

# ---- Image Processing Functions ----
def process_user_uploaded_image(image_bytes: bytes, query: str, model_name: str, project_id: str, location: str, credentials) -> str:
    """Process user uploaded image and generate response"""
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        
        mime_type = "image/jpeg"
        if image_bytes.startswith(b'\x89PNG'):
            mime_type = "image/png"
        elif image_bytes.startswith(b'GIF'):
            mime_type = "image/gif"
        elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
            mime_type = "image/webp"
        
        image_part = Part.from_data(image_bytes, mime_type=mime_type)
        
        prompt = f"""Analyze this image and answer the user's question: {query}

If this appears to be a screenshot or document related to HBS systems, provide detailed analysis. If it's not related to HBS, politely explain that you specialize in HBS system assistance."""
        
        response = model.generate_content([prompt, image_part])
        return response.text if response.text else "I couldn't analyze the image. Please try again."
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# ---- Semantic Analysis Functions ----
def analyze_user_sentiment_and_intent(query: str, conversation_context: str, model_name: str, project_id: str, location: str, credentials) -> Dict:
    """Use LLM to semantically analyze user sentiment and intent"""
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        
        # Keep analysis prompt lean
        context_snippet = truncate_to_token_limit(conversation_context, 500)
        
        analysis_prompt = f"""Analyze the user's query semantically and provide comprehensive analysis.

CONVERSATION CONTEXT:
{context_snippet}

USER QUERY: {query}

Analyze and classify:

1. INTENT: What is the user trying to accomplish?
   - "greeting" - Hello, hi, good morning, etc.
   - "farewell" - Goodbye, bye, see you later, etc.
   - "question" - Asking for information
   - "troubleshooting" - Having problems, things not working
   - "clarification" - Needs more explanation, doesn't understand
   - "alternative" - Asking for different methods/approaches
   - "escalation" - Wants human help, frustrated, urgent
   - "conversational" - Casual chat, compliments, small talk

2. SENTIMENT: What is the user's emotional state?
   - "positive" - Happy, satisfied, grateful
   - "neutral" - Matter-of-fact, informational
   - "frustrated" - Annoyed, confused, struggling
   - "urgent" - Pressed for time, critical issue
   - "negative" - Angry, disappointed, upset

3. CONTEXT_RELEVANCE: How does this relate to previous conversation?
   - "follow_up" - Building on previous topic
   - "clarification" - Asking for more details on previous answer
   - "correction" - Pointing out errors or inconsistencies
   - "new_topic" - Completely new subject
   - "continuation" - Same topic, different aspect

4. ESCALATION_NEEDED: Does this indicate need for human assistance?
   - Consider frustration level, complexity, repeated failures, explicit requests

Respond with ONLY a JSON object:
{{
    "intent": "one_of_the_intents_above",
    "sentiment": "one_of_the_sentiments_above", 
    "context_relevance": "one_of_the_relevance_types_above",
    "escalation_needed": true/false,
    "confidence": 0.95,
    "reasoning": "brief explanation of the analysis"
}}"""

        response = model.generate_content(
            analysis_prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=300,
                top_p=0.8,
                top_k=40
            )
        )
        
        if response.text:
            try:
                result = json.loads(response.text.strip())
                return result
            except json.JSONDecodeError:
                return {
                    "intent": "question",
                    "sentiment": "neutral",
                    "context_relevance": "new_topic",
                    "escalation_needed": False,
                    "confidence": 0.5,
                    "reasoning": "Failed to parse analysis response"
                }
        else:
            return {
                "intent": "question",
                "sentiment": "neutral", 
                "context_relevance": "new_topic",
                "escalation_needed": False,
                "confidence": 0.5,
                "reasoning": "No response from analysis model"
            }
    
    except Exception as e:
        return {
            "intent": "question",
            "sentiment": "neutral",
            "context_relevance": "new_topic", 
            "escalation_needed": False,
            "confidence": 0.3,
            "reasoning": f"Analysis error: {str(e)}"
        }

def generate_semantic_response(query: str, context_chunks: List[Dict], user_analysis: Dict, conversation_context: str, model_name: str, project_id: str, location: str, credentials) -> str:
    """Generate response using semantic understanding with optimized context"""
    
    # Build optimized context
    context_text = build_optimized_context(context_chunks, max_tokens=MAX_CONTEXT_TOKENS)
    
    # Limit conversation context
    context_section = ""
    if conversation_context:
        limited_conv_context = truncate_to_token_limit(conversation_context, 2000)
        context_section = f"""
RECENT CONVERSATION CONTEXT:
{limited_conv_context}

"""
    
    # Build user analysis section
    analysis_section = f"""
USER ANALYSIS:
- Intent: {user_analysis.get('intent', 'unknown')}
- Sentiment: {user_analysis.get('sentiment', 'neutral')}
- Context Relevance: {user_analysis.get('context_relevance', 'new_topic')}
- Escalation Needed: {user_analysis.get('escalation_needed', False)}
- Confidence: {user_analysis.get('confidence', 0):.2f}
- Reasoning: {user_analysis.get('reasoning', 'N/A')}

"""
    
    system_prompt = f"""You are an expert HBS (Help Business System) assistant with deep understanding of user intent and sentiment.

{context_section}{analysis_section}KNOWLEDGE BASE CONTEXT:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. **UNDERSTAND THE USER**: Consider their intent, sentiment, and context relevance
2. **RESPOND APPROPRIATELY**: 
   - If sentiment is "frustrated": Be extra patient and helpful
   - If intent is "troubleshooting": Provide step-by-step guidance
   - If intent is "clarification": Explain in simpler terms with more detail
   - If intent is "correction": Acknowledge the error and provide correct information
   - If intent is "escalation": Offer to connect with human support
   - If sentiment is "urgent": Prioritize quick, actionable solutions

3. **CONTEXT AWARENESS**:
   - If "follow_up": Build on previous conversation
   - If "clarification": Address specific points from previous answer
   - If "correction": Fix any inconsistencies or errors
   - If "new_topic": Start fresh but acknowledge context

4. **RESPONSE QUALITY**:
   - Be empathetic to user's emotional state
   - Provide specific, actionable information
   - If no relevant context found, be honest about limitations
   - Always be helpful and professional
   - Use appropriate tone based on user sentiment

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

def escalate_to_live_agent(query: str, conversation_context: str, user_analysis: Dict) -> str:
    """Escalate to live agent when bot cannot help"""
    
    confidence = user_analysis.get('confidence', 0)
    
    conversation_summary = f"""
CONVERSATION SUMMARY FOR LIVE AGENT:
===============================

USER'S CURRENT QUESTION: {query}

CONVERSATION CONTEXT:
{conversation_context}

USER ANALYSIS:
- Intent: {user_analysis.get('intent', 'unknown')}
- Sentiment: {user_analysis.get('sentiment', 'neutral')}
- Context Relevance: {user_analysis.get('context_relevance', 'new_topic')}
- Confidence: {confidence:.2f}
- Reasoning: {user_analysis.get('reasoning', 'N/A')}

ESCALATION REASON: Bot determined user needs human assistance based on semantic analysis.

===============================
"""
    
    if "escalation_requests" not in st.session_state:
        st.session_state.escalation_requests = []
    
    st.session_state.escalation_requests.append({
        "timestamp": len(st.session_state.messages),
        "query": query,
        "conversation_summary": conversation_summary,
        "user_analysis": user_analysis
    })
    
    response_parts = [
        "I understand you need additional assistance. Let me connect you with a live HBS support agent who can provide more specialized help.",
        "",
        "**Live Agent Escalation Request Submitted**",
        "",
        f"Your question: {query}",
        "",
        "A live agent will review your request and respond within 24 hours. In the meantime, you can:",
        "",
        "1. **Try rephrasing your question** with more specific details",
        "2. **Check the HBS documentation** for related procedures", 
        "3. **Contact HBS support directly** at support@hbs.com",
        "",
        f"**Reference ID:** ESC-{len(st.session_state.messages):04d}",
        "",
        "Thank you for using the HBS Help Chatbot!"
    ]
    
    return "\n".join(response_parts)

# ---- LangChain Integration ----
@st.cache_resource
def get_langchain_llm(project_id: str, location: str, _credentials, model_name: str):
    """Get LangChain LLM instance"""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    try:
        vertexai_init(project=project_id, location=location, credentials=_credentials)
        
        llm = VertexAI(
            model_name=model_name,
            project=project_id,
            location=location,
            temperature=0.1,
            max_output_tokens=2048,
            top_p=0.8,
            top_k=40
        )
        return llm
    except Exception as e:
        st.error(f"Error creating LangChain LLM: {e}")
        return None

@st.cache_resource
def get_conversation_chain(project_id: str, location: str, _credentials, model_name: str):
    """Get LangChain conversation chain with memory"""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    try:
        llm = get_langchain_llm(project_id, location, _credentials, model_name)
        if not llm:
            return None
        
        memory = ConversationBufferWindowMemory(
            k=6,
            return_messages=True
        )
        
        prompt = PromptTemplate(
            input_variables=["history", "input", "context", "user_analysis"],
            template="""You are an expert HBS (Help Business System) assistant with deep understanding of user intent and sentiment.

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER ANALYSIS:
{user_analysis}

CONVERSATION HISTORY:
{history}

USER QUESTION: {input}

INSTRUCTIONS:
1. **UNDERSTAND THE USER**: Consider their intent, sentiment, and context relevance
2. **RESPOND APPROPRIATELY**: 
   - If sentiment is "frustrated": Be extra patient and helpful
   - If intent is "troubleshooting": Provide step-by-step guidance
   - If intent is "clarification": Explain in simpler terms with more detail
   - If intent is "correction": Acknowledge the error and provide correct information
   - If intent is "escalation": Offer to connect with human support
   - If sentiment is "urgent": Prioritize quick, actionable solutions

3. **CONTEXT AWARENESS**:
   - If "follow_up": Build on previous conversation
   - If "clarification": Address specific points from previous answer
   - If "correction": Fix any inconsistencies or errors
   - If "new_topic": Start fresh but acknowledge context

4. **RESPONSE QUALITY**:
   - Be empathetic to user's emotional state
   - Provide specific, actionable information
   - If no relevant context found, be honest about limitations
   - Always be helpful and professional
   - Use appropriate tone based on user sentiment

RESPONSE:"""
        )
        
        chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=False
        )
        
        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def generate_response_with_langchain(query: str, context_chunks: List[Dict], user_analysis: Dict, project_id: str, location: str, _credentials, model_name: str) -> str:
    """Generate response using LangChain with conversation memory"""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    try:
        chain = get_conversation_chain(project_id, location, _credentials, model_name)
        if not chain:
            return None
        
        context_text = build_optimized_context(context_chunks, max_tokens=MAX_CONTEXT_TOKENS)
        
        analysis_text = f"""Intent: {user_analysis.get('intent', 'unknown')}
Sentiment: {user_analysis.get('sentiment', 'neutral')}
Context Relevance: {user_analysis.get('context_relevance', 'new_topic')}
Escalation Needed: {user_analysis.get('escalation_needed', False)}
Confidence: {user_analysis.get('confidence', 0):.2f}
Reasoning: {user_analysis.get('reasoning', 'N/A')}"""
        
        response = chain.predict(
            input=query,
            context=context_text,
            user_analysis=analysis_text
        )
        
        return response
    except Exception as e:
        st.error(f"Error generating response with LangChain: {e}")
        return None

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
    if "use_langchain" not in st.session_state:
        st.session_state.use_langchain = LANGCHAIN_AVAILABLE
    if "escalation_requests" not in st.session_state:
        st.session_state.escalation_requests = []

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
        
        # Model selection
        st.subheader("Model Settings")
        st.session_state.model_name = st.selectbox(
            "Select Model",
            CANDIDATE_MODELS,
            index=0,
            key="model_select"
        )
        
        # Show escalation requests
        if st.session_state.escalation_requests:
            st.subheader("📞 Live Agent Requests")
            for i, req in enumerate(st.session_state.escalation_requests):
                with st.expander(f"Request #{i+1} - {req['query'][:50]}..."):
                    st.write(f"**Query:** {req['query']}")
                    st.write(f"**Intent:** {req['user_analysis'].get('intent', 'unknown') if req['user_analysis'] else 'unknown'}")
                    st.write(f"**Sentiment:** {req['user_analysis'].get('sentiment', 'unknown') if req['user_analysis'] else 'unknown'}")
                    st.write(f"**Reference ID:** ESC-{req['timestamp']:04d}")
        
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
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", key="clear_btn"):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    st.title("HBS Help Chatbot")
    
    # Display welcome message if no messages yet
    if not st.session_state.messages:
        st.info("Hi! How can I help you today?")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📄 Sources"):
                    for source in message["sources"][:3]:
                        source_name = source['source']
                        similarity = source['similarity_score']
                        rerank = source.get('rerank_score', 'N/A')
                        st.write(f"📄 {source_name} (sim: {similarity:.3f}, rerank: {rerank})")
    
    # Image upload section
    upload_key = f"image_uploader_{len(st.session_state.messages)}"
    uploaded_image = st.file_uploader(
        "📷 Upload Image for Analysis",
        type=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'],
        key=upload_key
    )
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about HBS systems..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if uploaded_image is not None:
            with st.spinner("Analyzing your image..."):
                image_bytes = uploaded_image.read()
                response = process_user_uploaded_image(
                    image_bytes, 
                    prompt, 
                    st.session_state.model_name,
                    st.session_state.project_id,
                    st.session_state.location,
                    st.session_state.creds
                )
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": len(st.session_state.messages)
                })
            
            st.rerun()
        else:
            # Regular text processing with optimized retrieval
            conversation_context = ""
            if len(st.session_state.messages) > 1:
                conversation_context = get_conversation_context(st.session_state.messages)
            
            with st.spinner("Understanding your request..."):
                user_analysis = analyze_user_sentiment_and_intent(
                    prompt,
                    conversation_context,
                    st.session_state.model_name,
                    st.session_state.project_id,
                    st.session_state.location,
                    st.session_state.creds
                )
            
            if user_analysis.get('escalation_needed', False):
                response = escalate_to_live_agent(prompt, conversation_context, user_analysis)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": len(st.session_state.messages)
                })
            else:
                with st.spinner("Searching knowledge base..."):
                    context_chunks = search_index(
                        prompt, 
                        st.session_state.index, 
                        st.session_state.corpus,
                        st.session_state.project_id,
                        st.session_state.location,
                        st.session_state.creds,
                        st.session_state.model_name,
                        k=MAX_CHUNKS_FINAL,
                        min_similarity=0.3
                    )
                    
                    if st.session_state.use_langchain and LANGCHAIN_AVAILABLE:
                        response = generate_response_with_langchain(
                            prompt,
                            context_chunks,
                            user_analysis,
                            st.session_state.project_id,
                            st.session_state.location,
                            st.session_state.creds,
                            st.session_state.model_name
                        )
                        
                        if not response:
                            response = generate_semantic_response(
                                prompt,
                                context_chunks,
                                user_analysis,
                                conversation_context,
                                st.session_state.model_name,
                                st.session_state.project_id,
                                st.session_state.location,
                                st.session_state.creds
                            )
                    else:
                        response = generate_semantic_response(
                            prompt,
                            context_chunks,
                            user_analysis,
                            conversation_context,
                            st.session_state.model_name,
                            st.session_state.project_id,
                            st.session_state.location,
                            st.session_state.creds
                        )
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": context_chunks,
                        "timestamp": len(st.session_state.messages)
                    })
        
        st.rerun()

if __name__ == "__main__":
    main()
