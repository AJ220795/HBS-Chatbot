# Add this function after the existing functions
def escalate_to_live_agent(query: str, conversation_context: str, user_intent: Dict = None) -> str:
    """Escalate to live agent when bot cannot help"""
    
    # Create a summary of the conversation for the live agent
    conversation_summary = f"""
CONVERSATION SUMMARY FOR LIVE AGENT:
===============================

USER'S CURRENT QUESTION: {query}

CONVERSATION CONTEXT:
{conversation_context}

USER INTENT: {user_intent.get('intent', 'unknown') if user_intent else 'unknown'}
CONFIDENCE: {user_intent.get('confidence', 0):.2f if user_intent else 0}
REASONING: {user_intent.get('reasoning', 'N/A') if user_intent else 'N/A'}

ESCALATION REASON: Bot was unable to provide a satisfactory answer to the user's question.

===============================
"""
    
    # Store the escalation request in session state for the live agent
    if "escalation_requests" not in st.session_state:
        st.session_state.escalation_requests = []
    
    st.session_state.escalation_requests.append({
        "timestamp": len(st.session_state.messages),
        "query": query,
        "conversation_summary": conversation_summary,
        "user_intent": user_intent
    })
    
    return f"""I apologize, but I'm unable to provide a satisfactory answer to your question. Let me connect you with a live HBS support agent who can provide more specialized assistance.

**Live Agent Escalation Request Submitted**

Your question: "{query}"

A live agent will review your request and respond within 24 hours. In the meantime, you can:

1. **Try rephrasing your question** with more specific details
2. **Check the HBS documentation** for related procedures
3. **Contact HBS support directly** at support@hbs.com

**Reference ID:** ESC-{len(st.session_state.messages):04d}

Thank you for using the HBS Help Chatbot!"""

def should_escalate_to_agent(query: str, context_chunks: List[Dict], user_intent: Dict = None, conversation_context: str = "") -> bool:
    """Determine if the query should be escalated to a live agent"""
    
    # Check if we have relevant context
    if not context_chunks:
        # If no context and user is asking for help, escalate
        if user_intent and user_intent.get("intent") in ["troubleshooting", "clarification"]:
            return True
    
    # Check for escalation keywords
    escalation_keywords = [
        "escalate", "live agent", "human", "speak to someone", "talk to agent",
        "not working", "broken", "error", "bug", "issue", "problem",
        "urgent", "emergency", "critical", "asap"
    ]
    
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in escalation_keywords):
        return True
    
    # Check if user has asked the same question multiple times
    if len(st.session_state.messages) > 4:
        recent_questions = [msg["content"] for msg in st.session_state.messages[-6:] if msg["role"] == "user"]
        if len(set(recent_questions)) <= 2:  # User asking similar questions repeatedly
            return True
    
    return False

# Update the generate_response function
def generate_response(query: str, context_chunks: List[Dict], model_name: str, project_id: str, location: str, credentials, conversation_context: str = "", user_intent: Dict = None) -> str:
    """Generate response using Gemini with improved prompting and conversation context"""
    
    # Check if we should escalate to live agent
    if should_escalate_to_agent(query, context_chunks, user_intent, conversation_context):
        return escalate_to_live_agent(query, conversation_context, user_intent)
    
    if not context_chunks:
        # Handle different intents when no relevant context is found
        if user_intent and user_intent.get("intent") in ["troubleshooting", "clarification", "alternative"]:
            intent = user_intent["intent"]
            
            if intent == "troubleshooting":
                return "I understand you're having trouble with the steps I provided. Let me help troubleshoot this. Can you tell me:\n\n1. Which specific step are you stuck on?\n2. What exactly happens when you try to follow the instructions?\n3. Are you seeing any error messages?\n\nThis will help me provide more targeted assistance."
            
            elif intent == "clarification":
                return "I'd be happy to clarify! Based on our previous conversation, could you let me know which specific part you'd like me to explain in more detail? I can break it down step by step or use simpler terms."
            
            elif intent == "alternative":
                return "I don't have information about alternative methods for that topic in my knowledge base. Based on what I've shared, the main approach is what I described earlier. If you have a specific aspect you'd like to explore further, please let me know!"
        
        return "I don't have information about that topic in my knowledge base. Could you please rephrase your question or ask about HBS reports, procedures, or system features?"
    
    # Build context from retrieved chunks
    context_text = "\n\n".join([
        f"Source: {chunk['source']}\nContent: {chunk['text']}"
        for chunk in context_chunks
    ])
    
    # Check if we have structured data that can answer the question directly
    structured_answers = []
    for chunk in context_chunks:
        if chunk.get('content_type') == 'structured_data' and 'structured_data' in chunk:
            data = chunk['structured_data']
            # Check if this structured data can answer the query
            query_lower = query.lower()
            if any(field in query_lower for field in ['stock', 'contract', 'customer', 'days overdue', 'serial', 'make', 'model']):
                if 'customer_name' in data and 'customer' in query_lower:
                    if data['customer_name'].lower() in query_lower:
                        structured_answers.append(f"Customer: {data['customer_name']}, Contract: {data.get('contract', 'N/A')}, Stock: {data.get('stock_number', 'N/A')}, Make: {data.get('make', 'N/A')}, Model: {data.get('model', 'N/A')}, Days Overdue: {data.get('days_overdue', 'N/A')}, Serial: {data.get('serial', 'N/A')}")
                elif 'stock_number' in data and 'stock' in query_lower:
                    if data['stock_number'] in query:
                        structured_answers.append(f"Stock: {data['stock_number']}, Make: {data.get('make', 'N/A')}, Model: {data.get('model', 'N/A')}, Customer: {data.get('customer_name', 'N/A')}, Contract: {data.get('contract', 'N/A')}, Days Overdue: {data.get('days_overdue', 'N/A')}")
    
    # If we have direct structured answers, use them
    if structured_answers:
        return "\n\n".join(structured_answers)
    
    # Build conversation context for the prompt
    context_section = ""
    if conversation_context:
        context_section = f"""
RECENT CONVERSATION CONTEXT:
{conversation_context}

"""
    
    # Add intent information to the prompt
    intent_section = ""
    if user_intent and user_intent.get("intent") in ["troubleshooting", "clarification", "alternative"]:
        intent_section = f"""
USER INTENT: {user_intent['intent']} (confidence: {user_intent.get('confidence', 0):.2f})
REASONING: {user_intent.get('reasoning', '')}

"""
    
    # Improved system prompt with conversation context and intent awareness
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

# Update the main function to include escalation tracking
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
    if "pending_message" not in st.session_state:
        st.session_state.pending_message = None
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
        
        # LangChain toggle
        if LANGCHAIN_AVAILABLE:
            st.session_state.use_langchain = st.checkbox(
                "Use LangChain (Better Memory)",
                value=st.session_state.use_langchain,
                key="langchain_toggle"
            )
        else:
            st.info("LangChain not available")
        
        # Show escalation requests
        if st.session_state.escalation_requests:
            st.subheader("📞 Live Agent Requests")
            for i, req in enumerate(st.session_state.escalation_requests):
                with st.expander(f"Request #{i+1} - {req['query'][:50]}..."):
                    st.write(f"**Query:** {req['query']}")
                    st.write(f"**Intent:** {req['user_intent'].get('intent', 'unknown') if req['user_intent'] else 'unknown'}")
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
        st.info("Hi! How can I help you?")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📄 Sources"):
                    for source in message["sources"][:2]:
                        source_name = source['source']
                        similarity = source['similarity_score']
                        st.write(f"📄 {source_name} (similarity: {similarity:.3f})")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about HBS systems..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Check if this is a conversational query first
        conversational_response = get_conversational_response(prompt)
        
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
                        prompt,
                        conversation_context,
                        st.session_state.model_name,
                        st.session_state.project_id,
                        st.session_state.location,
                        st.session_state.creds
                    )
            
            # Search for relevant context
            with st.spinner("Thinking..."):
                context_chunks = search_index(
                    prompt, 
                    st.session_state.index, 
                    st.session_state.corpus,
                    st.session_state.project_id,
                    st.session_state.location,
                    st.session_state.creds,
                    k=2,
                    min_similarity=0.5
                )
                
                # Generate response
                if st.session_state.use_langchain and LANGCHAIN_AVAILABLE:
                    # Try LangChain first
                    response = generate_response_with_langchain(
                        prompt,
                        context_chunks,
                        st.session_state.project_id,
                        st.session_state.location,
                        st.session_state.creds,
                        st.session_state.model_name
                    )
                    
                    # Fallback to regular response if LangChain fails
                    if not response:
                        response = generate_response(
                            prompt,
                            context_chunks,
                            st.session_state.model_name,
                            st.session_state.project_id,
                            st.session_state.location,
                            st.session_state.creds,
                            conversation_context,
                            user_intent
                        )
                else:
                    # Use regular response generation
                    response = generate_response(
                        prompt,
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
        
        # Rerun to update the chat display
        st.rerun()

if __name__ == "__main__":
    main()
