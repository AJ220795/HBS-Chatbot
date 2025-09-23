# Add this debug version to your app.py - replace the existing process_kb_files() function

def process_kb_files() -> List[Dict]:
    """Process all KB files and create corpus with image data extraction - DEBUG VERSION"""
    corpus = []
    
    if not KB_DIR.exists():
        st.error(f"KB_DIR does not exist: {KB_DIR}")
        return corpus
    
    st.write(f"🔍 DEBUG: KB_DIR exists: {KB_DIR}")
    
    # List all files in KB directory
    files = list(KB_DIR.iterdir())
    st.write(f"�� DEBUG: Found {len(files)} files in KB directory:")
    for f in files:
        st.write(f"  - {f.name} ({f.suffix})")
    
    for file_path in files:
        if file_path.is_file():
            st.write(f"🔍 DEBUG: Processing {file_path.name}")
            try:
                if file_path.suffix.lower() == '.docx':
                    st.write(f"  �� Processing DOCX: {file_path.name}")
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
                        st.write(f"  ✅ Added {len(chunks)} chunks from DOCX")
                    else:
                        st.write(f"  ⚠️ No text extracted from DOCX")
                
                elif file_path.suffix.lower() == '.pdf':
                    st.write(f"  📄 Processing PDF: {file_path.name}")
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
                        st.write(f"  ✅ Added {len(chunks)} chunks from PDF")
                    else:
                        st.write(f"  ⚠️ No text extracted from PDF")
                
                elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
                    st.write(f"  🖼️ Processing IMAGE: {file_path.name}")
                    
                    # Check file size
                    file_size = file_path.stat().st_size
                    st.write(f"  📊 File size: {file_size} bytes")
                    
                    if file_size == 0:
                        st.write(f"  ❌ Image file is empty")
                        continue
                    
                    # Extract OCR text
                    try:
                        ocr_text = extract_text_from_image_bytes(file_path.read_bytes())
                        st.write(f"  🔤 OCR extracted {len(ocr_text)} characters")
                        st.write(f"  📝 First 200 chars: {ocr_text[:200]}...")
                        
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
                            st.write(f"  ✅ Added {len(chunks)} OCR chunks")
                            
                            # Parse structured data from OCR
                            try:
                                structured_data = parse_report_data_from_ocr(ocr_text, file_path.name)
                                st.write(f"  🔍 Parsed {len(structured_data)} structured data entries")
                                
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
                                
                                st.write(f"  ✅ Added {len(structured_data)} structured data entries")
                                
                            except Exception as e:
                                st.write(f"  ❌ Error parsing structured data: {e}")
                        else:
                            st.write(f"  ⚠️ No OCR text extracted from image")
                            
                    except Exception as e:
                        st.write(f"  ❌ Error extracting OCR: {e}")
                        import traceback
                        st.write(f"  📋 Traceback: {traceback.format_exc()}")
                
                else:
                    st.write(f"  ⚠️ Skipping unsupported file type: {file_path.suffix}")
                
            except Exception as e:
                st.error(f"❌ Error processing {file_path.name}: {e}")
                import traceback
                st.write(f"�� Traceback: {traceback.format_exc()}")
    
    st.write(f"🔍 DEBUG: Total corpus entries created: {len(corpus)}")
    return corpus
