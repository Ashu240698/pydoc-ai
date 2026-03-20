"""
Streamlit UI for PyDoc AI
"""

import streamlit as st

# 🔥 PAGE CONFIG MUST BE FIRST
st.set_page_config(
    page_title="PyDoc AI",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔥 DOWNLOAD INDEXES (cached to run only once)
@st.cache_resource
def ensure_indexes_downloaded():
    """Download indexes if needed (cached to run only once)."""
    import download_indexes
    download_indexes.download_indexes_if_needed()
    return True

# Download indexes immediately
ensure_indexes_downloaded()

# Now import LLM handler
from src.llm import LLMHandler

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


def display_sources(sources):
    """Display sources in a nice format."""
    st.markdown("### 📚 Sources")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i}: {source['source']} - {source['module']}", expanded=False):
            st.markdown(f"**Score:** {source['score']:.2f}")
            st.markdown(f"**URL:** [{source['url']}]({source['url']})")


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">🐍 PyDoc AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your AI-Powered Python Documentation Assistant</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## About")
        st.markdown("""
        PyDoc AI helps you find answers to Python questions using:
        - 🔍 Hybrid search (BM25 + FAISS)
        - 🎯 Cross-encoder re-ranking
        - 🤖 AI-powered responses
        - 📚 Official documentation sources
        """)
        
        st.markdown("---")
        
        st.markdown("### 🔑 Configuration")
        st.markdown("PyDoc AI uses Groq's LLM API")
        
        # API Key management
        if 'api_key_entered' not in st.session_state:
            st.session_state.api_key_entered = False
        
        if not st.session_state.api_key_entered:
            user_api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Enter your Groq API key",
                key="api_key_input"
            )
            
            st.markdown("""
            **Don't have a key?**
            1. Visit [console.groq.com](https://console.groq.com)
            2. Sign up (free)
            3. Create an API key
            4. Paste it above
            
            ✅ Free tier: 30 requests/minute
            """)
            
            if user_api_key:
                st.session_state.user_api_key = user_api_key
                st.session_state.api_key_entered = True
                st.rerun()
            else:
                st.warning("⚠️ Enter your API key to start")
                st.stop()
        else:
            st.success("✅ API Key configured")
            if st.button("🔄 Change API Key", use_container_width=True):
                st.session_state.api_key_entered = False
                if 'user_api_key' in st.session_state:
                    del st.session_state.user_api_key
                if 'handler' in st.session_state:
                    del st.session_state.handler
                st.rerun()
            
            user_api_key = st.session_state.user_api_key
        
        st.markdown("---")
        
        st.markdown("## Settings")
        k_results = st.slider("Number of sources to retrieve", 2, 8, 4)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            if 'handler' in st.session_state:
                st.session_state.handler.conversation_history = []
            st.success("Conversation cleared!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Example Questions")
        st.markdown("""
        - How do I read a CSV file?
        - What is the requests library?
        - How to use pandas DataFrame?
        - How do I handle JSON in Python?
        """)
    
    # Initialize handler with user's API key
    if 'handler' not in st.session_state:
        with st.spinner("Loading PyDoc AI..."):
            st.session_state.handler = LLMHandler(api_key=user_api_key)
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a Python question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                response_text, sources = st.session_state.handler.query(
                    prompt,
                    k=k_results,
                    include_history=True
                )
            
            # Display response
            st.markdown(response_text)
            
            # Display sources
            display_sources(sources)
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "sources": sources
        })


if __name__ == "__main__":
    main()