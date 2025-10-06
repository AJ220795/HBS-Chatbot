import os
import io
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict

import streamlit as st
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

# LangChain imports (no UI at import time)
try:
    from langchain_community.llms import VertexAI
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import ConversationChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ---- App constants ----
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
KB_DIR = APP_DIR / "kb"
EXTRACT_DIR = DATA_DIR / "kb_extracted"
INDEX_PATH = DATA_DIR / "faiss.index"
CORPUS_PATH = DATA_DIR / "corpus.json"
META_PATH = DATA_DIR / "kb_meta.json"
EMB_CACHE_PATH = DATA_DIR / "embeddings_cache.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite"
]

DEFAULT_LOCATION = "us-central1"

# ---- Helpers (hashing, IO) ----
def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def sha256_text(t: str) -> str:
    return sha256_bytes(t.encode("utf-8"))

def read_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def write_json(path: Path, obj):
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        st.error(f"Failed to write {path.name}: {e}")

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
            token_est = sum(max(1, len(x) // 4) for x in buf)
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
        if re.match(r'^[A-Z\s]+$', line) and len(line) > 3:  # Customer name pattern
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

# ---- Embeddings / Index with Cache ----
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

def build_faiss_index_with_cache(corpus: List[Dict], project_id: str, location: str, credentials, emb_cache: Dict) -> faiss.IndexFlatIP:
    """
    Incremental embedding builder:
      - Reuses cached embeddings when chunk text (hash) is unchanged
      - Embeds only missing chunks
      - Builds new FAISS index from all chunk vectors
    """
    # Prepare keys & find what's missing
    keys = []
    to_embed_texts = []
    to_embed_keys = []

    for item in corpus:
        txt = item.get("text", "")
        chunk_hash = sha256_text(txt)[:12]
        key = f"{item.get('source','')}|{item.get('chunk_id','')}|{chunk_hash}"
        keys.append((key, txt))

        if key not in emb_cache:
            to_embed_keys.append(key)
            to_embed_texts.append(txt)

    # Batch-embed missing
    if to_embed_texts:
        arr = embed_texts(to_embed_texts, project_id, location, credentials)
        if arr.size == 0:
            st.error("Failed to embed new chunks.")
        else:
            for k, vec in zip(to_embed_keys, arr.tolist()):
                emb_cache[k] = vec

    # Assemble all vectors in corpus order
    vectors = []
    for key, txt in keys:
        vec = emb_cache.get(key)
        if vec is None:
            # Fallback: embed single (should be rare)
            arr = embed_texts([txt], project_id, location, credentials)
            if arr.size:
                vec = arr[0].tolist()
                emb_cache[key] = vec
            else:
                vec = [0.0] * 768  # default dim; adjust if model changes
        vectors.append(vec)

    vectors_np = np.array(vectors, dtype=np.float32)
    # Normalize for cosine
    faiss.normalize_L2(vectors_np)

    # Build index
    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)

    return index

# ---- Corpus building: Incremental (B) ----
def process_kb_files_incremental(prev_corpus: List[Dict], prev_meta: Dict) -> (List[Dict], Dict, Dict):
    """
    Builds a new corpus incrementally:
      - Reuses prev_corpus entries for unchanged files
      - Re-extracts text/OCR and re-chunks for new/modified files
      - Drops entries for deleted files
    Returns: (new_corpus, new_meta, telemetry_stats)
    """
    telemetry = {
        "files_total": 0,
        "files_new": 0,
        "files_modified": 0,
        "files_unchanged": 0,
        "files_deleted": 0,
        "chunks_total": 0,
        "ocr_failures": 0,
        "structured_parse_failures": 0
    }

    prev_by_source = {}
    for item in prev_corpus or []:
        prev_by_source.setdefault(item.get("source", ""), []).append(item)

    new_corpus: List[Dict] = []
    new_meta: Dict = {}

    # Scan current files
    if not KB_DIR.exists():
        st.error(f"KB_DIR does not exist: {KB_DIR}")
        return [], {}, telemetry

    files = [p for p in KB_DIR.iterdir() if p.is_file()]
    telemetry["files_total"] = len(files)

    found_sources = set()

    for file_path in files:
        fname = file_path.name
        found_sources.add(fname)
        try:
            raw = file_path.read_bytes()
            fhash = sha256_bytes(raw)
            new_meta[fname] = {"sha256": fhash, "mtime": file_path.stat().st_mtime}

            prev = prev_meta.get(fname)
            if prev and prev.get("sha256") == fhash:
                # Unchanged: reuse previous chunks
                telemetry["files_unchanged"] += 1
                for item in prev_by_source.get(fname, []):
                    new_corpus.append(item)
                continue

            # New or modified: reprocess
            is_new = fname not in prev_meta
            if is_new:
                telemetry["files_new"] += 1
            else:
                telemetry["files_modified"] += 1

            if file_path.suffix.lower() == ".docx":
                text = extract_text_from_docx_bytes(raw)

                if text.strip():
                    chunks = chunk_text(text)
                    for i, chunk in enumerate(chunks):
                        new_corpus.append({
                            "text": chunk,
                            "source": fname,
                            "chunk_id": i,
                            "file_type": file_path.suffix.lower()
                        })

            elif file_path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf_bytes(raw)
                if text.strip():
                    chunks = chunk_text(text)
                    for i, chunk in enumerate(chunks):
                        new_corpus.append({
                            "text": chunk,
                            "source": fname,
                            "chunk_id": i,
                            "file_type": file_path.suffix.lower()
                        })

            elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
                try:
                    ocr_text = extract_text_from_image_bytes(raw)
                    if ocr_text.strip():
                        chunks = chunk_text(ocr_text)
                        for i, chunk in enumerate(chunks):
                            new_corpus.append({
                                "text": chunk,
                                "source": fname,
                                "chunk_id": i,
                                "file_type": file_path.suffix.lower(),
                                "content_type": "ocr_text"
                            })

                        try:
                            structured_data = parse_report_data_from_ocr(ocr_text, fname)
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

                                new_corpus.append({
                                    "text": searchable_text,
                                    "source": fname,
                                    "chunk_id": len(new_corpus),
                                    "file_type": file_path.suffix.lower(),
                                    "content_type": "structured_data",
                                    "structured_data": data_item
                                })
                        except Exception as e:
                            telemetry["structured_parse_failures"] += 1
                            st.warning(f"Structured parse failed for {fname}: {e}")
                except Exception as e:
                    telemetry["ocr_failures"] += 1
                    st.warning(f"OCR failed for {fname}: {e}")

        except Exception as e:
            st.error(f"Error processing {fname}: {e}")

    # Deleted files: present in prev_meta but not now
    for fname in prev_meta.keys():
        if fname not in found_sources:
            telemetry["files_deleted"] += 1

    telemetry["chunks_total"] = len(new_corpus)
    return new_corpus, new_meta, telemetry

# ---- Simple conversational / context ----
def get_conversational_response(query: str) -> str:
    query_lower = query.lower().strip()

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
    farewells = ["bye", "goodbye", "see you", "farewell", "thanks", "thank you", "ttyl", "talk to you later"]
    casual = ["what's up", "how are you", "how's it going", "what's new", "how do you do"]
    vague_responses = ["sure", "ok", "okay", "yes", "yep", "yeah", "alright", "fine", "good"]
    compliments = ["good job", "well done", "excellent", "great", "awesome", "amazing"]

    if any(g in query_lower for g in greetings):
        return "Hi! How can I help you?"
    if any(f in query_lower for f in farewells):
        return "Goodbye! Feel free to come back anytime if you have more questions about HBS systems."
    if any(c in query_lower for c in casual):
        return "I'm doing well, thank you! I'm here to help you with any questions about HBS systems, reports, or procedures. What can I assist you with today?"
    if any(v in query_lower for v in vague_responses):
        return "I'm ready to help! What can I assist you with regarding the HBS system? Please let me know what you need."
    if any(cmp in query_lower for cmp in compliments):
        return "Thank you! I'm here to help with any HBS system questions you might have."
    return None

def get_conversation_context(messages: List[Dict], max_context: int = 3) -> str:
    if len(messages) < 2:
        return ""
    recent = messages[-max_context*2:]  # last 3 exchanges
    parts = []
    for msg in recent:
        if msg["role"] == "user":
            parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            content = msg['content'].split('\n\n')[0]
            parts.append(f"Assistant: {content}")
    return "\n".join(parts)

def classify_user_intent(query: str, conversation_context: str, model_name: str, project_id: str, location: str, credentials) -> Dict:
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)

        classification_prompt = f"""Analyze the user's query and classify their intent. Consider the conversation context.

CONVERSATION CONTEXT:
{conversation_context}

USER QUERY: {query}

Classify the user's intent into one of these categories:

1. "troubleshooting" - User is having problems following previous instructions, things aren't working, can't find something, getting errors, etc.
2. "clarification" - User wants more explanation, doesn't understand something, wants simpler terms, more details, etc.
3. "alternative" - User is asking for other methods, alternatives, different approaches, etc.
4. "new_question" - User is asking a completely new question unrelated to previous conversation
5. "conversational" - Simple greetings, farewells, casual chat

Respond with ONLY a JSON object in this exact format:
{{
    "intent": "one_of_the_categories_above",
    "confidence": 0.95,
    "reasoning": "brief explanation of why this classification was chosen"
}}

Be precise and consider the semantic meaning, not just keywords."""

        response = model.generate_content(
            classification_prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=200,
                top_p=0.8,
                top_k=40
            )
        )

        if response.text:
            try:
                return json.loads(response.text.strip())
            except json.JSONDecodeError:
                return {"intent": "new_question", "confidence": 0.5, "reasoning": "Failed to parse classification response"}
        return {"intent": "new_question", "confidence": 0.5, "reasoning": "No response from classification model"}
    except Exception as e:
        return {"intent": "new_question", "confidence": 0.3, "reasoning": f"Classification error: {str(e)}"}

def generate_response(query: str, context_chunks: List[Dict], model_name: str, project_id: str, location: str, credentials, conversation_context: str = "", user_intent: Dict = None) -> str:
    if not context_chunks:
        if user_intent and user_intent.get("intent") in ["troubleshooting", "clarification", "alternative"]:
            intent = user_intent["intent"]
            if intent == "troubleshooting":
                return ("I understand you're having trouble with the steps I provided. Let me help troubleshoot this. "
                        "Can you tell me:\n\n1. Which specific step are you stuck on?\n2. What exactly happens when you try to follow the instructions?\n"
                        "3. Are you seeing any error messages?\n\nThis will help me provide more targeted assistance.")
            elif intent == "clarification":
                return ("I'd be happy to clarify! Based on our previous conversation, could you let me know which specific part you'd like me to explain in more detail? "
                        "I can break it down step by step or use simpler terms.")
            elif intent == "alternative":
                return ("I don't have information about alternative methods for that topic in my knowledge base. Based on what I've shared, the main approach is what I described earlier. "
                        "If you have a specific aspect you'd like to explore further, please let me know!")
        return "I don't have information about that topic in my knowledge base. Could you please rephrase your question or ask about HBS reports, procedures, or system features?"

    context_text = "\n\n".join([f"Source: {c['source']}\nContent: {c['text']}" for c in context_chunks])

    structured_answers = []
    for c in context_chunks:
        if c.get('content_type') == 'structured_data' and 'structured_data' in c:
            data = c['structured_data']
            query_lower = query.lower()
            if any(field in query_lower for field in ['stock', 'contract', 'customer', 'days overdue', 'serial', 'make', 'model']):
                if 'customer_name' in data and 'customer' in query_lower and data['customer_name'].lower() in query_lower:
                    structured_answers.append(
                        f"Customer: {data['customer_name']}, Contract: {data.get('contract','N/A')}, Stock: {data.get('stock_number','N/A')}, "
                        f"Make: {data.get('make','N/A')}, Model: {data.get('model','N/A')}, Days Overdue: {data.get('days_overdue','N/A')}, Serial: {data.get('serial','N/A')}"
                    )
                elif 'stock_number' in data and 'stock' in query_lower and data['stock_number'] in query:
                    structured_answers.append(
                        f"Stock: {data['stock_number']}, Make: {data.get('make','N/A')}, Model: {data.get('model','N/A')}, "
                        f"Customer: {data.get('customer_name','N/A')}, Contract: {data.get('contract','N/A')}, Days Overdue: {data.get('days_overdue','N/A')}"
                    )
    if structured_answers:
        return "\n\n".join(structured_answers)

    context_section = f"\nRECENT CONVERSATION CONTEXT:\n{conversation_context}\n\n" if conversation_context else ""
    intent_section = ""
    if user_intent and user_intent.get("intent") in ["troubleshooting", "clarification", "alternative"]:
        intent_section = (f"\nUSER INTENT: {user_intent['intent']} (confidence: {user_intent.get('confidence', 0):.2f})\n"
                          f"REASONING: {user_intent.get('reasoning','')}\n\n")

    system_prompt = f"""You are an expert HBS (Help Business System) assistant. You have access to detailed documentation about HBS systems, reports, and procedures.

{context_section}{intent_section}CONTEXT FROM KNOWLEDGE BASE:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. **FIRST**: Check if the context above is actually relevant to the user's question
2. If the context is NOT relevant to the question, say "I don't have specific information about that topic in my knowledge base. Could you please rephrase your question or ask about HBS reports, procedures, or system features?"
3. If the context IS relevant, answer the user's question using ONLY the information provided
4. **IMPORTANT**: Consider the user's intent:
   - If intent is "troubleshooting": Provide step-by-step troubleshooting guidance
   - If intent is "clarification": Explain in simpler terms with more detail
   - If intent is "alternative": Acknowledge the request and explain limitations
5. Be specific and detailed in your response
6. If the context contains step-by-step procedures, provide them clearly
7. If the context mentions specific options, fields, or settings, list them exactly
8. If you need to make assumptions, state them clearly
9. If the context doesn't contain enough information to fully answer the question, say "I don't have enough information to answer that question based on my resources" and suggest what additional information might be needed
10. If the question is unclear or ambiguous, ask clarifying questions
11. Always be helpful and professional

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

def generate_image_response(query: str, image_bytes: bytes, model_name: str, project_id: str, location: str, credentials, mime_type: str = "image/jpeg") -> str:
    try:
        vertexai_init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)
        image_part = Part.from_data(image_bytes, mime_type=mime_type)
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
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", key="clear_btn"):
            st.session_state.messages = []
            st.rerun()

    # Create custom HTML chat interface
    chat_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                height: 100vh;
                overflow: hidden;
            }}
            
            .chat-container {{
                display: flex;
                flex-direction: column;
                height: 100vh;
                background: #f8f9fa;
            }}
            
            .chat-header {{
                background: white;
                padding: 1rem;
                border-bottom: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                z-index: 1000;
            }}
            
            .chat-messages {{
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }}
            
            .message {{
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 18px;
                word-wrap: break-word;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .message-user {{
                background: #007bff;
                color: white;
                margin-left: auto;
                border-radius: 18px 18px 5px 18px;
            }}
            
            .message-assistant {{
                background: white;
                color: #333;
                margin-right: auto;
                border-radius: 18px 18px 18px 5px;
                border: 1px solid #e0e0e0;
            }}
            
            .sources {{
                background: #e3f2fd;
                padding: 8px 12px;
                border-radius: 8px;
                margin-top: 8px;
                font-size: 0.9em;
                border-left: 3px solid #2196f3;
            }}
            
            .chat-input-container {{
                background: white;
                padding: 1rem;
                border-top: 1px solid #e0e0e0;
                display: flex;
                gap: 1rem;
                align-items: center;
            }}
            
            .chat-input {{
                flex: 1;
                padding: 12px 20px;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                font-size: 16px;
                outline: none;
            }}
            
            .chat-input:focus {{
                border-color: #007bff;
                box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
            }}
            
            .send-button {{
                background: #007bff;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
            }}
            
            .send-button:hover {{
                background: #0056b3;
            }}
            
            .upload-button {{
                background: #6c757d;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 50%;
                cursor: pointer;
                width: 48px;
                height: 48px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>HBS Help Chatbot</h1>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message message-assistant">Hi! How can I help you?</div>
    """
    
    # Add existing messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            chat_html += f'<div class="message message-user">{message["content"]}</div>'
        else:
            chat_html += f'<div class="message message-assistant">{message["content"]}</div>'
            if "sources" in message and message["sources"]:
                chat_html += '<div class="sources"><strong>Sources:</strong><br>'
                for source in message["sources"][:2]:
                    source_name = source['source']
                    similarity = source['similarity_score']
                    chat_html += f'📄 {source_name} (similarity: {similarity:.3f})<br>'
                chat_html += '</div>'
    
    chat_html += """
            </div>
            
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="chatInput" placeholder="Ask me anything about HBS systems..." />
                <button class="upload-button" onclick="document.getElementById('fileInput').click()">📷</button>
                <button class="send-button" onclick="sendMessage()">Send</button>
                <input type="file" id="fileInput" style="display: none" accept="image/*" />
            </div>
        </div>
        
        <script>
            function sendMessage() {
                const input = document.getElementById('chatInput');
                const message = input.value.trim();
                if (message) {
                    // Add user message to chat
                    addMessage(message, 'user');
                    input.value = '';
                    
                    // Send to Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: message
                    }, '*');
                }
            }
            
            function addMessage(text, role) {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message message-${role}`;
                messageDiv.textContent = text;
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            // Handle Enter key
            document.getElementById('chatInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Auto-scroll to bottom
            function scrollToBottom() {
                const chatMessages = document.getElementById('chatMessages');
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            // Scroll to bottom when page loads
            window.addEventListener('load', scrollToBottom);
        </script>
    </body>
    </html>
    """
    
    # Display the custom chat interface
    st.components.v1.html(chat_html, height=600)
    
    # Handle messages from the custom interface
    if st.session_state.get('user_message'):
        user_message = st.session_state.user_message
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Check if this is a conversational query first
        conversational_response = get_conversational_response(user_message)
        
        if conversational_response:
            # For conversational queries, don't search KB or show sources
            st.session_state.messages.append({
                "role": "assistant", 
                "content": conversational_response,
                "timestamp": len(st.session_state.messages)
            })
        else:
            # Get conversation context
            conversation_context = get_conversation_context(st.session_state.messages)
            
            # Classify user intent using LLM
            user_intent = None
            if conversation_context:
                with st.spinner("Understanding your request..."):
                    user_intent = classify_user_intent(
                        user_message,
                        conversation_context,
                        st.session_state.model_name,
                        st.session_state.project_id,
                        st.session_state.location,
                        st.session_state.creds
                    )
            
            # Search for relevant context
            with st.spinner("Thinking..."):
                context_chunks = search_index(
                    user_message, 
                    st.session_state.index, 
                    st.session_state.corpus,
                    st.session_state.project_id,
                    st.session_state.location,
                    st.session_state.creds,
                    k=2,
                    min_similarity=0.5
                )
                
                # Generate response
                response = generate_response(
                    user_message,
                    context_chunks,
                    st.session_state.model_name,
                    st.session_state.project_id,
                    st.session_state.location,
                    st.session_state.creds,
                    conversation_context,
                    user_intent
                )
                
                # Add assistant response to messages
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": context_chunks,
                    "timestamp": len(st.session_state.messages)
                })
        
        # Clear the user message
        st.session_state.user_message = None
        st.rerun()
        
if __name__ == "__main__":
    main()
