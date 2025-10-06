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
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import VertexAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage

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
    "gemini-2.5-pro"
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
    
    # Clean up OCR text
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    
    # Look for report patterns
    if "overdue" in filename.lower() or "overdue" in ocr_text.lower():
        # Parse Overdue Equipment Report
        structured_data.extend(parse_overdue_report(ocr_text, filename))
    elif "outbound" in filename.lower() or "outbound" in ocr_text.lower():
        # Parse Outbound Report
        structured_data.extend(parse_outbound_report(ocr_text, filename))
    elif "equipment list" in filename.lower() or "equipment list" in ocr_text.lower():
        # Parse Equipment List Report
        structured_data.extend(parse_equipment_list_report(ocr_text, filename))
    
    return structured_data

def parse_overdue_report(ocr_text: str, filename: str) -> List[Dict]:
    """Parse Overdue Equipment Report data"""
    data = []
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    
    # Look for customer data patterns
    for i, line in enumerate(lines):
        # Pattern: Customer name, Contract, Phone, Stock, Make, Model, Type, Year, Serial, Date Out, Expected, Days Over
        if re.match(r'^[A-Z\s]+$', line) and len(line) > 3:  # Customer name pattern
            # Look for the next lines that contain the data
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # Extract contract number (C followed by digits and R)
                contract_match = re.search(r'C\d+R', next_line)
                if contract_match:
                    customer_name = line
                    contract = contract_match.group()
                    
                    # Try to extract other data from surrounding lines
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
                    
                    # Look in the next few lines for data
                    for j in range(i, min(i + 5, len(lines))):
                        current_line = lines[j]
                        
                        # Extract phone number
                        phone_match = re.search(r'\(\d{3}\)\s*\d{3}-\d{4}', current_line)
                        if phone_match:
                            phone = phone_match.group()
                        
                        # Extract stock number (5 digits)
                        stock_match = re.search(r'\b\d{5}\b', current_line)
                        if stock_match:
                            stock = stock_match.group()
                        
                        # Extract make (BOB, KUB, etc.)
                        make_match = re.search(r'\b(BOB|KUB|JD|BOM)\b', current_line)
                        if make_match:
                            make = make_match.group()
                        
                        # Extract model
                        model_match = re.search(r'\b(T650|E32|E42|U55-4R3AP|U35-4R3A|E26|KX080R3AT3|KX121R3TA|KX121RRATS|211D-50|690B|442)\b', current_line)
                        if model_match:
                            model = model_match.group()
                        
                        # Extract equipment type
                        type_match = re.search(r'\b(SKIDSTEER|EXCAVATOR|ROLLER)\b', current_line)
                        if type_match:
                            equipment_type = type_match.group()
                        
                        # Extract year
                        year_match = re.search(r'\b(2013|2014|2015|2016|1979|2006|2008|2012)\b', current_line)
                        if year_match:
                            year = year_match.group()
                        
                        # Extract serial number
                        serial_match = re.search(r'\b[A-Z0-9]{6,}\b', current_line)
                        if serial_match and len(serial_match.group()) > 6:
                            serial = serial_match.group()
                        
                        # Extract dates
                        date_match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', current_line)
                        if date_match:
                            if not date_out:
                                date_out = date_match.group()
                            else:
                                expected = date_match.group()
                        
                        # Extract days overdue
                        days_match = re.search(r'\b\d{1,4}\b', current_line)
                        if days_match and days_match.group().isdigit():
                            days_over = days_match.group()
                    
                    # Create structured data entry
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
    
    # Look for customer data patterns
    for i, line in enumerate(lines):
        if re.match(r'^[A-Z\s]+$', line) and len(line) > 3:  # Customer name pattern
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                contract_match = re.search(r'C\d+R', next_line)
                if contract_match:
                    customer_name = line
                    contract = contract_match.group()
                    
                    # Extract other data from surrounding lines
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
                        
                        # Extract date/time
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
    
    # Look for equipment data patterns
    for i, line in enumerate(lines):
        # Look for stock number patterns
        stock_match = re.search(r'\b\d{5}\b', line)
        if stock_match:
            stock = stock_match.group()
            
            # Extract other data from the same line or nearby lines
            make = ""
            model = ""
            equipment_type = ""
            year = ""
            serial = ""
            location = ""
            meter = ""
            
            # Look for make
            make_match = re.search(r'\b(BOB|KUB|JD|BOM)\b', line)
            if make_match:
                make = make_match.group()
            
            # Look for model
            model_match = re.search(r'\b(T650|E32|E42|U55-4R3AP|U35-4R3A|E26|KX080R3AT3|KX121R3TA|KX121RRATS|211D-50|690B|442)\b', line)
            if model_match:
                model = model_match.group()
            
            # Look for equipment type
            type_match = re.search(r'\b(SKIDSTEER|EXCAVATOR|ROLLER)\b', line)
            if type_match:
                equipment_type = type_match.group()
            
            # Look for year
            year_match = re.search(r'\b(2013|2014|2015|2016|1979|2006|2008|2012)\b', line)
            if year_match:
                year = year_match.group()
            
            # Look for serial
            serial_match = re.search(r'\b[A-Z0-9]{6,}\b', line)
            if serial_match and len(serial_match.group()) > 6:
                serial = serial_match.group()
            
            # Look for meter reading
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

def expand_query(query: str) -> str:
    """Expand query with related terms for better search"""
    query_lower = query.lower()
    
    # Add related terms based on common HBS topics
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

def search_index(query: str, index, corpus: List[Dict], project_id: str, location: str, credentials, k: int = 2, min_similarity: float = 0.5) -> List[Dict]:
    """Search FAISS index with query expansion and relevance threshold"""
    if index is None or not corpus:
        return []
    
    try:
        # Expand query for better search
        expanded_query = expand_query(query)
        
        # Generate query embedding
        query_embeddings = embed_texts([expanded_query], project_id, location, credentials)
        if query_embeddings.size == 0:
            return []
        
        # Normalize query embedding
        faiss.normalize_L2(query_embeddings)
        
        # Search - get more results to filter
        scores, indices = index.search(query_embeddings, min(10, len(corpus)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(corpus) and score >= min_similarity:  # Only include relevant results
                results.append({
                    **corpus[idx],
                    "similarity_score": float(score)
                })
                if len(results) >= k:  # Stop when we have enough good results
                    break
        
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
    """Process all KB files and create corpus with image data extraction"""
    corpus = []
    
    if not KB_DIR.exists():
        st.error(f"KB_DIR does not exist: {KB_DIR}")
        return corpus
    
    # List all files in KB directory
    files = list(KB_DIR.iterdir())
    
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
                    # Check file size
                    file_size = file_path.stat().st_size
                    if file_size == 0:
                        continue
                    
                    # Extract OCR text
                    try:
                        ocr_text = extract_text_from_image_bytes(file_path.read_bytes())
                        
                        if ocr_text.strip():
                            # Add raw OCR text as chunks
                            chunks = chunk_text(ocr_text)
                            for i, chunk in enumerate(chunks):
                                corpus.append({
                                    "text": chunk,
                                    "source": file_path.name,
                                    "chunk_id": i,
                                    "file_type": file_path.suffix.lower(),
                                    "content_type": "ocr_text"
                                })
                            
                            # Parse structured data from OCR
                            try:
                                structured_data = parse_report_data_from_ocr(ocr_text, file_path.name)
                                
                                for data_item in structured_data:
                                    # Create searchable text from structured data
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
    
    return corpus

def get_conversational_response(query: str) -> str:
    """Handle simple conversational queries"""
    query_lower = query.lower().strip()
    
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
    farewells = ["bye", "goodbye", "see you", "farewell", "thanks", "thank you", "ttyl", "talk to you later"]
    casual = ["what's up", "how are you", "how's it going", "what's new", "how do you do"]
    vague_responses = ["sure", "ok", "okay", "yes", "yep", "yeah", "alright", "fine", "good"]
    compliments = ["good job", "well done", "excellent", "great", "awesome", "amazing"]
    
    if any(greeting in query_lower for greeting in greetings):
        return "Hi! How can I help you?"
    
    if any(farewell in query_lower for farewell in farewells):
        return "Goodbye! Feel free to come back anytime if you have more questions about HBS systems."
    
    if any(casual_phrase in query_lower for casual_phrase in casual):
        return "I'm doing well, thank you! I'm here to help you with any questions about HBS systems, reports, or procedures. What can I assist you with today?"
    
    if any(vague in query_lower for vague in vague_responses):
        return "I'm ready to help! What can I assist you with regarding the HBS system? Please let me know what you need."
    
    if any(compliment in query_lower for compliment in compliments):
        return "Thank you! I'm here to help with any HBS system questions you might have."
    
    return None

def initialize_langchain_components(project_id: str, location: str, credentials, model_name: str):
    """Initialize LangChain components"""
    try:
        # Initialize Vertex AI
        vertexai_init(project=project_id, location=location, credentials=credentials)
        
        # Initialize LangChain LLM
        llm = VertexAI(
            model_name=model_name,
            temperature=0.1,
            max_output_tokens=2048,
            top_p=0.8,
            top_k=40
        )
        
        # Initialize conversation memory
        memory = ConversationBufferWindowMemory(
            k=3,  # Keep last 3 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Create conversation chain with custom prompt
        prompt_template = """You are an expert HBS (Help Business System) assistant. You have access to detailed documentation about HBS systems, reports, and procedures.

CONTEXT FROM KNOWLEDGE BASE:
{context}

CHAT HISTORY:
{chat_history}

USER QUESTION: {input}

INSTRUCTIONS:
1. **FIRST**: Check if the context above is actually relevant to the user's question
2. If the context is NOT relevant to the question, say "I don't have specific information about that topic in my knowledge base. Could you please rephrase your question or ask about HBS reports, procedures, or system features?"
3. If the context IS relevant, answer the user's question using ONLY the information provided
4. **IMPORTANT**: Consider the conversation history and user's intent:
   - If the user is having trouble following previous instructions, provide step-by-step troubleshooting guidance
   - If the user wants clarification, explain in simpler terms with more detail
   - If the user is asking for alternatives, acknowledge the request and explain limitations
5. Be specific and detailed in your response
6. If the context contains step-by-step procedures, provide them clearly
7. If the context mentions specific options, fields, or settings, list them exactly
8. If you need to make assumptions, state them clearly
9. If the context doesn't contain enough information to fully answer the question, say "I don't have enough information to answer that question based on my resources" and suggest what additional information might be needed
10. If the question is unclear or ambiguous, ask clarifying questions
11. Always be helpful and professional

RESPONSE:"""

        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "input"],
            template=prompt_template
        )
        
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )
        
        return conversation, memory
        
    except Exception as e:
        st.error(f"Error initializing LangChain components: {e}")
        return None, None

def generate_response_with_langchain(query: str, context_chunks: List[Dict], conversation, memory) -> str:
    """Generate response using LangChain with conversation memory"""
    try:
        # Build context from retrieved chunks
        context_text = ""
        if context_chunks:
            context_text = "\n\n".join([
                f"Source: {chunk['source']}\nContent: {chunk['text']}"
                for chunk in context_chunks
            ])
        
        # Use LangChain conversation chain
        response = conversation.predict(
            context=context_text,
            input=query
        )
        
        return response if response else "I couldn't generate a response. Please try rephrasing your question."
    
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
        st.session_state.model_name = CANDIDATE_MODELS[0]  # Now defaults to gemini-2.5-flash-lite
    if "kb_loaded" not in st.session_state:
        st.session_state.kb_loaded = False
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "memory" not in st.session_state:
        st.session_state.memory = None

    # Initialize credentials
    try:
        sa_info = json.loads(st.secrets["google"]["credentials_json"])
        st.session_state.creds = service_account.Credentials.from_service_account_info(sa_info)
        st.session_state.project_id = st.secrets["google"]["project"]
        st.session_state.location = st.secrets["google"].get("location", DEFAULT_LOCATION)
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        st.stop()

    # Initialize LangChain components
    if st.session_state.conversation is None:
        with st.spinner("Initializing LangChain components..."):
            conversation, memory = initialize_langchain_components(
                st.session_state.project_id,
                st.session_state.location,
                st.session_state.creds,
                st.session_state.model_name
            )
            st.session_state.conversation = conversation
            st.session_state.memory = memory

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
        new_model = st.selectbox(
            "Select Model",
            CANDIDATE_MODELS,
            index=0,
            key="model_select"
        )
        
        # Reinitialize LangChain if model changed
        if new_model != st.session_state.model_name:
            st.session_state.model_name = new_model
            with st.spinner("Reinitializing with new model..."):
                conversation, memory = initialize_langchain_components(
                    st.session_state.project_id,
                    st.session_state.location,
                    st.session_state.creds,
                    st.session_state.model_name
                )
                st.session_state.conversation = conversation
                st.session_state.memory = memory
                st.rerun()
        
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
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.rerun()

    # Main chat interface
    st.title("HBS Help Chatbot")
    
    # Show initial greeting if no messages
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("Hi! How can I help you?")
    
    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"][:2]):  # Limit to 2 sources
                        # Create clickable source link
                        source_name = source['source']
                        similarity = source['similarity_score']
                        content_preview = source['text'][:200] + "..."
                        
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
                                    key=f"download_{i}_{message.get('timestamp', 0)}"
                                )
                            else:
                                st.write(f"**{source_name}** (similarity: {similarity:.3f})")
                        except Exception:
                            st.write(f"**{source_name}** (similarity: {similarity:.3f})")
                        
                        st.write(content_preview)
                        st.write("---")

    # Chat input
    if prompt := st.chat_input("Ask me anything about HBS systems..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if this is a conversational query first
        conversational_response = get_conversational_response(prompt)
        
        with st.chat_message("assistant"):
            if conversational_response:
                # For conversational queries, don't search KB or show sources
                st.markdown(conversational_response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": conversational_response,
                    "timestamp": len(st.session_state.messages)
                })
                
                # Add to LangChain memory
                if st.session_state.memory:
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(conversational_response)
            else:
                # Search for relevant context
                with st.spinner("Thinking..."):
                    context_chunks = search_index(
                        prompt, 
                        st.session_state.index, 
                        st.session_state.corpus,
                        st.session_state.project_id,
                        st.session_state.location,
                        st.session_state.creds,
                        k=2,  # Limit to 2 sources
                        min_similarity=0.5  # Increased threshold to 0.5
                    )
                    
                    # Generate response with LangChain
                    response = generate_response_with_langchain(
                        prompt,
                        context_chunks,
                        st.session_state.conversation,
                        st.session_state.memory
                    )
                    
                    st.markdown(response)
                    
                    # Show sources if available - limit to 2
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
                                        st.write(f"**{source_name}** (similarity: {similarity:.3f})")
                                except Exception:
                                    st.write(f"**{source_name}** (similarity: {similarity:.3f})")
                                
                                st.write(content_preview)
                                st.write("---")
                    
                    # Add assistant message with sources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": context_chunks,
                        "timestamp": len(st.session_state.messages)
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
