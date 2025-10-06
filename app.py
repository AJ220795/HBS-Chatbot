            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📄 Sources"):
                    for source in message["sources"][:2]:
                        source_name = source['source']
                        similarity = source['similarity_score']
                        
                        # Make sources clickable
                        if st.button(f"📄 {source_name} (similarity: {similarity:.3f})", key=f"source_{source_name}_{hash(source['text'])}"):
                            display_document_content(source_name, source['text'])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about HBS systems..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get conversation context
        conversation_context = ""
        if len(st.session_state.messages) > 1:
            conversation_context = get_conversation_context(st.session_state.messages)
        
        # Analyze user sentiment and intent semantically
        with st.spinner("Understanding your request..."):
            user_analysis = analyze_user_sentiment_and_intent(
                prompt,
                conversation_context,
                st.session_state.model_name,
                st.session_state.project_id,
                st.session_state.location,
                st.session_state.creds
            )
        
        # Check if escalation is needed
        if user_analysis.get('escalation_needed', False):
            response = escalate_to_live_agent(prompt, conversation_context, user_analysis)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "timestamp": len(st.session_state.messages)
            })
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
                    k=2,
                    min_similarity=0.5
                )
                
                # Generate response
                if st.session_state.use_langchain and LANGCHAIN_AVAILABLE:
                    # Try LangChain first
                    response = generate_response_with_langchain(
                        prompt,
                        context_chunks,
                        user_analysis,
                        st.session_state.project_id,
                        st.session_state.location,
                        st.session_state.creds,
                        st.session_state.model_name
                    )
                    
                    # Fallback to regular response if LangChain fails
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
                    # Use regular semantic response generation
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
