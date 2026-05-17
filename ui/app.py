import streamlit as st
import requests
import uuid
import os

# Backend API URL
API_URL = os.getenv("API_URL", "http://localhost:8000/api/chat")

st.set_page_config(page_title="Banking AI Copilot", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    /* User chat alignment to the right */
    div[data-testid="stChatMessage"]:has(.user-msg-marker) {
        flex-direction: row-reverse;
        text-align: right;
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 15px 15px 0px 15px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant chat alignment to the left */
    div[data-testid="stChatMessage"]:has(.assistant-msg-marker) {
        background-color: var(--background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 15px 15px 15px 0px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Adjust avatar margins for user */
    div[data-testid="stChatMessage"]:has(.user-msg-marker) > div:first-child {
        margin-left: 1rem;
        margin-right: 0;
    }

    /* Hide markers */
    .user-msg-marker, .assistant-msg-marker {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        if "last_trace" in st.session_state:
            del st.session_state["last_trace"]
        st.rerun()

    st.divider()
    st.subheader("🔍 CRAG Pipeline Trace")
    if "last_trace" in st.session_state and "steps" in st.session_state.last_trace:
        for i, step in enumerate(st.session_state.last_trace["steps"]):
            with st.expander(f"Step {i+1}: {step.get('action')}", expanded=True):
                for k, v in step.items():
                    if k != "action":
                        st.write(f"**{k}**: {v}")
    else:
        st.info("No trace available yet. Ask a question to see the CRAG pipeline execution steps.")

st.title("🏦 Banking AI Copilot (BFSI)")
st.caption("Powered by Self-Reflective RAG (CRAG) & LangChain")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout
col_chat = st.container()

with col_chat:
    st.subheader("Chat")
    
    # Create a container for the chat history
    chat_history_container = st.container()
    
    with chat_history_container:
        # Display chat messages from history (Standard order: Oldest to Newest)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown("<span class='user-msg-marker'></span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='assistant-msg-marker'></span>", unsafe_allow_html=True)
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if msg.get("confidence") == "low":
                        st.warning("⚠️ Low confidence — based on web search, not verified policy document.")
                    if msg.get("compliance") == "flagged":
                        st.error("🚨 Response flagged by compliance guardrails.")
                    elif "compliance" in msg:
                        st.success("✅ Compliance Check Passed")
                    
                    if msg.get("sources"):
                        st.caption("Sources: " + ", ".join(msg["sources"]))

    # Accept user input (Placed after the history container so it appears at the bottom)
    if prompt := st.chat_input("Ask a banking question (e.g., 'What are the home loan eligibility criteria?')"):
        
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Render the newly submitted message inside the chat container
        with chat_history_container:
            with st.chat_message("user"):
                st.markdown("<span class='user-msg-marker'></span>", unsafe_allow_html=True)
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st.markdown("<span class='assistant-msg-marker'></span>", unsafe_allow_html=True)
                with st.spinner("Thinking... (Running CRAG Pipeline)"):
                    try:
                        response = requests.post(API_URL, json={"query": prompt, "session_id": st.session_state.session_id})
                        response.raise_for_status()
                        data = response.json()
                        
                        answer = data.get("answer", "Error generating answer.")
                        confidence = data.get("confidence_level", "high")
                        compliance = data.get("compliance_status", "flagged")
                        sources = data.get("sources", [])
                        trace = data.get("trace", {})
                        
                        st.markdown(answer)
                        
                        if confidence == "low":
                            st.warning("⚠️ Low confidence — based on web search, not verified policy document.")
                        if compliance == "flagged":
                            st.error("🚨 Response flagged by compliance guardrails.")
                        else:
                            st.success("✅ Compliance Check Passed")
                            
                        if sources:
                            st.caption("Sources: " + ", ".join(sources))
                            
                        # Save trace for the side panel
                        st.session_state.last_trace = trace
                            
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "confidence": confidence,
                            "compliance": compliance,
                            "sources": sources
                        })
                        
                    except requests.exceptions.RequestException as e:
                        st.error(f"API Error: {e}")



