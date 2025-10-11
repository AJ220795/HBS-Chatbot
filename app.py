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

# Enhanced token management for large KB
MAX_CONTEXT_TOKENS = 150000  # Up to 150K tokens for context
MAX_CHUNKS_INITIAL = 50      # Retrieve 50 candidates initially
MAX_CHUNKS_FINAL = 10        # Keep top 10 after re-ranking

# ---- Helper Functions ----
def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 1 token ≈ 4 characters)"""
    return max(1, len(text) // 4)

def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit"""
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text
    
    # Truncate by character count
    max_chars = max_tokens * 4
    return text[:max_chars] + "..."

def sanitize_user_input(user_input: str) -> str:
    """Sanitize user input to prevent prompt injection"""
    dangerous_patterns = [
        r"ignore previous instructions",
        r"disregard all above",
        r"forget everything",
        r"new instructions:",
        r"system:",
        r"override",
        r"admin mode"
    ]
    
    sanitized = user_input
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()

def get_credentials():
    """Get Google Cloud credentials from Streamlit secrets"""
    try:
        credentials_dict = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        return credentials
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return None

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text: str, max_tokens: int = 300, overlap_sentences: int = 2) -> List[str]:
    """Improved chunking with much smaller chunks for better retrieval"""
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
    
    # Validate and split oversized chunks with very aggressive limits
    validated_chunks = []
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk)
        if chunk_tokens > 2000:  # Very low limit for safety
            # Split by words if still too large
            sub_chunks = split_oversized_chunk(chunk, max_tokens=2000)
            validated_chunks.extend(sub_chunks)
        else:
            validated_chunks.append(chunk)
    
    return validated_chunks

def split_oversized_chunk(chunk: str, max_tokens: int = 2000) -> List[str]:
    """Split an oversized chunk into smaller pieces by words"""
    words = chunk.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for word in words:
        word_tokens = max(1, len(word) // 4)
        if current_tokens + word_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(word)
        current_tokens += word_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [chunk]

# ---- OCR and Image Processing ----
def extract_text_from_image(image_path: Path) -> str:
    """Extract text from image using OCR"""
    try:
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh)
        return text.strip()
    except Exception as e:
        st.warning(f"OCR failed for {image_path.name}: {e}")
        return ""

def parse_overdue_equipment_report(ocr_text: str, filename: str) -> List[Dict]:
    """Parse Overdue Equipment Report data"""
    data = []
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    
    for i, line in enumerate(lines):
        contract_match = re.search(r'\b\d{5,}\b', line)
        if contract_match:
            contract = contract_match.group()
            
            customer_name = ""
            phone = ""
            stock = ""
            make = ""
            model = ""
            equipment_type = ""
            year = ""
            serial = ""
            date_time_out = ""
            
            for j in range(max(0, i-2), min(len(lines), i+3)):
                current_line = lines[j]
                
                name_match = re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', current_line)
                if name_match and not customer_name:
                    customer_name = name_match.group()
                
                phone_match = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', current_line)
                if phone_match:
                    phone = phone_match.group()
                
                stock_match = re.search(r'\b\d{5}\b', current_line)
                if stock_match and stock_match.group() != contract:
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
                    "report_type": "Overdue Equipment Report"
                })
    
    return data

def parse_outbound_report(ocr_text: str, filename: str) -> List[Dict]:
    """Parse Outbound Report data"""
    data = []
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    
    for i, line in enumerate(lines):
        contract_match = re.search(r'\b\d{5,}\b', line)
        if contract_match:
            contract = contract_match.group()
            
            customer_name = ""
            phone = ""
            stock = ""
            make = ""
            model = ""
            equipment_type = ""
            year = ""
            serial = ""
            date_time_out = ""
            
            for j in range(max(0, i-2), min(len(lines), i+3)):
                current_line = lines[j]
                
                name_match = re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', current_line)
                if name_match and not customer_name:
                    customer_name = name_match.group()
                
                phone_match = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', current_line)
                if phone_match:
                    phone = phone_match.group()
                
                stock_match = re.search(r'\b\d{5}\b', current_line)
                if stock_match and stock_match.group() != contract:
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

# ---- Document Processing ----
def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        st.warning(f"Error reading {file_path.name}: {e}")
        return ""

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        st.warning(f"Error reading {file_path.name}: {e}")
        return ""

# ---- Query Expansion ----
def expand_query(query: str) -> str:
    """Expand query with domain-specific synonyms"""
    query_lower = query.lower()
    
    if "buyback" in query_lower or "buy back" in query_lower:
        return f"{query} CNH buyback verified submission evaluation"
    elif "document" in query_lower or "upload" in query_lower or "file" in query_lower:
        return f"{query} document management DM NetView upload search"
    elif "transport" in query_lower or "ticket" in query_lower or "delivery" in query_lower:
        return f"{query} transport ticket TRNS delivery unit"
    elif "task" in query_lower or "overdue" in query_lower or "completed" in query_lower:
        return f"{query} NetView Insight tasks status"
    elif "rental" in query_lower or "rent" in query_lower:
        return f"{query} rental equipment outbound overdue"
    elif "serial" in query_lower:
        return f"{query} serial number equipment"
    else:
        return query

def rerank_chunks(query: str, chunks: List[Dict], model_name: str, project_id: str, location: str, credentials, top_k: int = 10) -> List[Dict]:
    """Use LLM to re-rank chunks by relevance to query"""
    if len(chunks) <= top_k:
        # Assign default rerank scores to all chunks
        for chunk in chunks:
            if 'rerank_score' not in chunk:
                chunk['rerank_score'] = chunk.get('similarity_score', 0.5)
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
                    
                    # Add default rerank scores to remaining chunks
                    for chunk in chunks[30:]:
                        if 'rerank_score' not in chunk:
                            chunk['rerank_score'] = chunk.get('similarity_score', 0.5)
                    
                    return sorted_chunks[:top_k]
            except:
                pass
        
        # Fallback: return top k by similarity with default rerank scores
        for chunk in chunks:
            if 'rerank_score' not in chunk:
                chunk['rerank_score'] = chunk.get('similarity_score', 0.5)
        return chunks[:top_k]
    
    except Exception as e:
        # Fallback: return top k by similarity with default rerank scores
        for chunk in chunks:
            if 'rerank_score' not in chunk:
                chunk['rerank_score'] = chunk.get('similarity_score', 0.5)
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

def search_index(query: str, index, corpus: List[Dict], project_id: str, location: str, credentials, model_name: str, k: int = MAX_CHUNKS_FINAL, min_similarity: float = 0.6) -> List[Dict]:
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
        st.error(f"Search error: {str(e)}")
        return []

# ---- Embedding and Indexing ----
def embed_texts(texts: List[str], project_id: str, location: str, credentials, batch_size: int = 250) -> np.ndarray:
    """Generate embeddings for texts in batches to handle large corpora"""
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        
        all_embeddings = []
        
        # Validate and filter texts before embedding with very strict limits
        valid_texts = []
        skipped_count = 0
        for i, text in enumerate(texts):
            token_count = estimate_tokens(text)
            if token_count > 10000:  # Very strict limit
                skipped_count += 1
                if skipped_count <= 5:  # Only show first 5 warnings to avoid spam
                    st.warning(f"Skipping text {i+1} with {token_count} tokens (exceeds 10K limit)")
                # Add a placeholder embedding of zeros
                all_embeddings.append(np.zeros(768))  # text-embedding-005 has 768 dimensions
            else:
                valid_texts.append((i, text))
        
        if skipped_count > 5:
            st.warning(f"Skipped {skipped_count} total texts that exceeded token limits")
        
        # Process valid texts in batches
        for batch_idx in range(0, len(valid_texts), batch_size):
            batch = valid_texts[batch_idx:batch_idx + batch_size]
            batch_texts = [text for _, text in batch]
            
            # Calculate total batch tokens
            batch_tokens = sum(estimate_tokens(text) for text in batch_texts)
            st.info(f"Processing embedding batch {batch_idx//batch_size + 1}/{(len(valid_texts) + batch_size - 1)//batch_size} (tokens: {batch_tokens})")
            
            batch_embeddings = model.get_embeddings(batch_texts)
            
            # Insert embeddings at correct positions
            for (original_idx, _), embedding in zip(batch, batch_embeddings):
                # Store embedding at its original position
                if len(all_embeddings) <= original_idx:
                    all_embeddings.extend([None] * (original_idx - len(all_embeddings) + 1))
                all_embeddings[original_idx] = embedding.values
        
        # Fill any remaining None values with zeros
        all_embeddings = [emb if emb is not None else np.zeros(768) for emb in all_embeddings]
        
        return np.array(all_embeddings).astype(np.float32)
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return np.array([])

@st.cache_resource
def build_faiss_index(corpus: List[Dict], _project_id: str, _location: str, _credentials) -> faiss.Index:
    """Build FAISS index from corpus"""
    if not corpus:
        return None
    
    texts = [item["text"] for item in corpus]
    embeddings = embed_texts(texts, _project_id, _location, _credentials)
    
    if embeddings.size == 0:
        st.error("Failed to build index")
        return None
    
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, str(INDEX_PATH))
    
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    
    return index

@st.cache_resource
def load_or_build_index(_project_id: str, _location: str, _credentials):
    """Load existing index or build new one"""
    if INDEX_PATH.exists() and CORPUS_PATH.exists():
        try:
            index = faiss.read_index(str(INDEX_PATH))
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            return index, corpus
        except Exception as e:
            st.warning(f"Could not load index: {e}. Rebuilding...")
    
    corpus = process_kb_files()
    index = build_faiss_index(corpus, _project_id, _location, _credentials)
    return index, corpus

def process_kb_files() -> List[Dict]:
    """Process all KB files and create corpus with image data extraction"""
    corpus = []
    
    if not KB_DIR.exists():
        st.error(f"KB_DIR does not exist: {KB_DIR}")
        return corpus
    
    files = list(KB_DIR.iterdir())
    st.info(f"Found {len(files)} files in KB directory")
    
    processed_files = 0
    total_chunks_before_validation = 0
    
    for file_path in files:
        if file_path.suffix.lower() == ".docx":
            text = extract_text_from_docx(file_path)
            if text:
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    corpus.append({
                        "text": chunk,
                        "source": file_path.name,
                        "chunk_id": f"{file_path.stem}_{i}",
                        "type": "document"
                    })
        
        elif file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
            if text:
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    corpus.append({
                        "text": chunk,
                        "source": file_path.name,
                        "chunk_id": f"{file_path.stem}_{i}",
                        "type": "document"
                    })
        
        elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            ocr_text = extract_text_from_image(file_path)
            if ocr_text:
                if "overdue" in file_path.name.lower():
                    structured_data = parse_overdue_equipment_report(ocr_text, file_path.name)
                    for data in structured_data:
                        text_repr = f"Overdue Equipment: {data['customer_name']} (Contract: {data['contract']}) - {data['make']} {data['model']} {data['equipment_type']} (Stock: {data['stock_number']}, Serial: {data['serial']}, Year: {data['year']}) - Out: {data['date_time_out']}"
                        corpus.append({
                            "text": text_repr,
                            "source": file_path.name,
                            "chunk_id": f"{file_path.stem}_{data['contract']}",
                            "type": "structured_data",
                            "data": data
                        })
                elif "outbound" in file_path.name.lower():
                    structured_data = parse_outbound_report(ocr_text, file_path.name)
                    for data in structured_data:
                        text_repr = f"Outbound Rental: {data['customer_name']} (Contract: {data['contract']}) - {data['make']} {data['model']} {data['equipment_type']} (Stock: {data['stock_number']}, Serial: {data['serial']}, Year: {data['year']}) - Out: {data['date_time_out']}"
                        corpus.append({
                            "text": text_repr,
                            "source": file_path.name,
                            "chunk_id": f"{file_path.stem}_{data['contract']}",
                            "type": "structured_data",
                            "data": data
                        })
                elif "equipment" in file_path.name.lower() and "list" in file_path.name.lower():
                    structured_data = parse_equipment_list_report(ocr_text, file_path.name)
                    for data in structured_data:
                        text_repr = f"Equipment: {data['make']} {data['model']} {data['equipment_type']} (Stock: {data['stock_number']}, Serial: {data['serial']}, Year: {data['year']}, Meter: {data['meter']})"
                        corpus.append({
                            "text": text_repr,
                            "source": file_path.name,
                            "chunk_id": f"{file_path.stem}_{data['stock_number']}",
                            "type": "structured_data",
                            "data": data
                        })
                else:
                    chunks = chunk_text(ocr_text)
                    for i, chunk in enumerate(chunks):
                        corpus.append({
                            "text": chunk,
                            "source": file_path.name,
                            "chunk_id": f"{file_path.stem}_{i}",
                            "type": "ocr_text"
                        })
        
        processed_files += 1
    
    # Validate chunk sizes and split oversized ones
    st.info("Validating chunk sizes...")
    validated_corpus = []
    oversized_chunks = 0
    max_chunk_size = 0
    total_chunks = len(corpus)
    
    for idx, item in enumerate(corpus):
        chunk_tokens = estimate_tokens(item["text"])
        max_chunk_size = max(max_chunk_size, chunk_tokens)
        
        if chunk_tokens > 2000:
            oversized_chunks += 1
            if oversized_chunks <= 3:  # Show details for first 3 oversized chunks
                st.warning(f"Chunk {idx+1} from {item['source']} has {chunk_tokens} tokens - splitting")
            
            # Split the oversized chunk
            sub_chunks = split_oversized_chunk(item["text"], max_tokens=2000)
            for i, sub_chunk in enumerate(sub_chunks):
                validated_corpus.append({
                    **item,
                    "text": sub_chunk,
                    "chunk_id": f"{item['chunk_id']}_split_{i}"
                })
        else:
            validated_corpus.append(item)
    
    if oversized_chunks > 0:
        st.warning(f"Split {oversized_chunks} oversized chunks into smaller pieces")
    
    st.success(f"Processed {processed_files} files, created {len(validated_corpus)} chunks (after validation)")
    return validated_corpus

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

4. ESCALATION_NEEDED: Does this require human intervention?
   - true if: user explicitly asks for human help, bot cannot answer adequately, urgent issue, frustrated user
   - false otherwise

5. CONFIDENCE: How confident are you in this analysis? (0.0 to 1.0)

6. REASONING: Brief explanation of your analysis

Return JSON with keys: intent, sentiment, context_relevance, escalation_needed, confidence, reasoning

Example: {{"intent": "question", "sentiment": "neutral", "context_relevance": "new_topic", "escalation_needed": false, "confidence": 0.9, "reasoning": "User asking straightforward informational question"}}

Return ONLY valid JSON, no other text."""

        response = model.generate_content(
            analysis_prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=500
            )
        )
        
        if response.text:
            try:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return analysis
            except json.JSONDecodeError:
                pass
        
        # Fallback analysis
        return {
            "intent": "question",
            "sentiment": "neutral",
            "context_relevance": "new_topic", 
            "escalation_needed": False,
            "confidence": 0.5,
            "reasoning": "Using fallback analysis"
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
    
    system_prompt = f"""You are an expert HBS (Help Business System) assistant. Provide concise, helpful answers.

{context_section}{analysis_section}KNOWLEDGE BASE CONTEXT:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. **BE CONCISE**: Give direct, actionable answers. Avoid unnecessary explanations.
2. **BE HELPFUL**: Focus on what the user needs to know to solve their problem.
3. **BE ACCURATE**: Only use information from the knowledge base context above.
4. **ESCALATION**: If user wants human help or you can't answer adequately, offer to connect them with an HBS Support Technician.

RESPONSE GUIDELINES:
- Keep responses under 200 words unless detailed steps are needed
- Use bullet points for multiple items
- Be specific about HBS features, modules, or processes
- If no relevant info found, say "I don't have specific information about that in my knowledge base. Would you like me to connect you with an HBS Support Technician?"

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
        "I understand you need additional assistance. Let me connect you with an HBS Support Technician.",
        "",
        "**Connecting you with an HBS Support Technician now...**",
        "",
        f"Your question: {query}",
        "",
        "**What to expect:**",
        "- An HBS Support Technician will join the chat shortly",
        "- They have access to additional resources and can provide specialized assistance",
        "- Feel free to ask any follow-up questions once connected",
        "",
        f"**Reference ID:** ESC-{len(st.session_state.messages):04d}",
        "",
        "Please hold while I connect you with a support technician..."
    ]
    
    return "\n".join(response_parts)

# ---- LangChain Integration ----
@st.cache_resource
def get_langchain_llm(_model_name: str, _project_id: str, _location: str, _credentials):
    """Get LangChain LLM instance"""
    vertexai_init(project=_project_id, location=_location, credentials=_credentials)
    return VertexAI(
        model_name=_model_name,
        temperature=0.7,
        max_output_tokens=1024,
        credentials=_credentials
    )

@st.cache_resource
def get_conversation_chain(_model_name: str, _project_id: str, _location: str, _credentials):
    """Get LangChain conversation chain"""
    llm = get_langchain_llm(_model_name, _project_id, _location, _credentials)
    
    memory = ConversationBufferWindowMemory(
        k=5,
        return_messages=True,
        memory_key="chat_history"
    )
    
    template = """You are an expert HBS (Help Business System) assistant. 
Use the conversation history to provide contextual, helpful responses.

Conversation History:
{chat_history}

Current Question: {input}

Provide a helpful, accurate response based on your HBS knowledge."""
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "input"],
        template=template
    )
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False
    )
    
    return conversation

def generate_response_with_langchain(query: str, context_chunks: List[Dict], model_name: str, project_id: str, location: str, credentials) -> str:
    """Generate response using LangChain"""
    try:
        conversation = get_conversation_chain(model_name, project_id, location, credentials)
        
        # Build optimized context
        context_text = build_optimized_context(context_chunks, max_tokens=MAX_CONTEXT_TOKENS)
        
        enhanced_query = f"""Based on the following HBS knowledge base context, answer the user's question.

CONTEXT:
{context_text}

QUESTION: {query}

Provide a helpful, accurate response."""
        
        response = conversation.predict(input=enhanced_query)
        return response
    
    except Exception as e:
        return f"Error with LangChain response: {str(e)}"

# ---- Streamlit App ----
def initialize_app():
    """Initialize the Streamlit app"""
    st.set_page_config(
        page_title="HBS Help Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "credentials" not in st.session_state:
        st.session_state.credentials = get_credentials()
    
    if "project_id" not in st.session_state:
        st.session_state.project_id = st.secrets.get("project_id", "")
    
    if "location" not in st.session_state:
        st.session_state.location = st.secrets.get("location", "us-central1")
    
    if "model_name" not in st.session_state:
        st.session_state.model_name = CANDIDATE_MODELS[0]
    
    if "index" not in st.session_state or "corpus" not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.index, st.session_state.corpus = load_or_build_index(
                st.session_state.project_id,
                st.session_state.location,
                st.session_state.credentials
            )

def main():
    initialize_app()
    
    st.title("🤖 HBS Help Chatbot")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.session_state.model_name = st.selectbox(
            "Select Model",
            CANDIDATE_MODELS,
            index=CANDIDATE_MODELS.index(st.session_state.model_name)
        )
        
        if st.button("🔄 Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔨 Rebuild Knowledge Base"):
            INDEX_PATH.unlink(missing_ok=True)
            CORPUS_PATH.unlink(missing_ok=True)
            st.cache_resource.clear()
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("📄 Sources"):
                    for i, source in enumerate(message["sources"]):
                        source_name = source.get('source', 'Unknown')
                        sim_score = source.get('similarity_score', 0)
                        rerank_score = source.get('rerank_score', 'N/A')
                        
                        # Format rerank score
                        if isinstance(rerank_score, (int, float)):
                            rerank_display = f"{rerank_score:.1f}"
                        else:
                            rerank_display = "N/A"
                        
                        with st.expander(f"📄 {source_name} (sim: {sim_score:.3f}, rerank: {rerank_display})"):
                            st.text_area(
                                "Content",
                                value=source.get('text', ''),
                                height=150,
                                key=f"source_{len(st.session_state.messages)}_{i}_{hash(source.get('text', ''))}"
                            )
    
    # Welcome message
    if not st.session_state.messages:
        st.info("Hi! How can I help you today?")
    
    # Image upload (above text input)
    uploaded_image = st.file_uploader(
        "📎 Upload an image (optional)",
        type=["png", "jpg", "jpeg"],
        key=f"image_uploader_{len(st.session_state.messages)}"
    )
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about HBS..."):
        # Sanitize input
        prompt = sanitize_user_input(prompt)
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Handle image if uploaded
                if uploaded_image is not None:
                    image_bytes = uploaded_image.read()
                    response = process_user_uploaded_image(
                        image_bytes,
                        prompt,
                        st.session_state.model_name,
                        st.session_state.project_id,
                        st.session_state.location,
                        st.session_state.credentials
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    # Get conversation context
                    conversation_context = get_conversation_context(st.session_state.messages)
                    
                    # Analyze user sentiment and intent
                    user_analysis = analyze_user_sentiment_and_intent(
                        prompt,
                        conversation_context,
                        st.session_state.model_name,
                        st.session_state.project_id,
                        st.session_state.location,
                        st.session_state.credentials
                    )
                    
                    # Check if escalation is needed
                    if user_analysis.get('escalation_needed', False):
                        response = escalate_to_live_agent(prompt, conversation_context, user_analysis)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        # Search knowledge base
                        context_chunks = search_index(
                            prompt,
                            st.session_state.index,
                            st.session_state.corpus,
                            st.session_state.project_id,
                            st.session_state.location,
                            st.session_state.credentials,
                            st.session_state.model_name
                        )
                        
                        # Generate semantic response
                        response = generate_semantic_response(
                            prompt,
                            context_chunks,
                            user_analysis,
                            conversation_context,
                            st.session_state.model_name,
                            st.session_state.project_id,
                            st.session_state.location,
                            st.session_state.credentials
                        )
                        
                        st.markdown(response)
                        
                        # Store message with sources
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": context_chunks
                        })
                        
                        # Display sources
                        if context_chunks:
                            with st.expander("📄 Sources"):
                                for i, source in enumerate(context_chunks):
                                    source_name = source.get('source', 'Unknown')
                                    sim_score = source.get('similarity_score', 0)
                                    rerank_score = source.get('rerank_score', 'N/A')
                                    
                                    # Format rerank score
                                    if isinstance(rerank_score, (int, float)):
                                        rerank_display = f"{rerank_score:.1f}"
                                    else:
                                        rerank_display = "N/A"
                                    
                                    with st.expander(f"📄 {source_name} (sim: {sim_score:.3f}, rerank: {rerank_display})"):
                                        st.text_area(
                                            "Content",
                                            value=source.get('text', ''),
                                            height=150,
                                            key=f"source_new_{i}_{hash(source.get('text', ''))}"
                                        )

if __name__ == "__main__":
    main()
