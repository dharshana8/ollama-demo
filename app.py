import streamlit as st
from main import load_dataset, build_vector_db, stream_response, get_sources

st.set_page_config(
    page_title="Cat Facts Chatbot",
    page_icon="🐱",
    layout="centered",
)

st.markdown("""
<style>
    .source-card {
        background: #f9f5ff;
        border-left: 3px solid #7c3aed;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #374151;
    }
    .score-badge {
        display: inline-block;
        background: #7c3aed;
        color: white;
        border-radius: 20px;
        padding: 1px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .stat-box {
        background: #f3f0ff;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🐱 Loading cat facts into vector DB...")
def init():
    dataset = load_dataset()
    build_vector_db(dataset)
    return len(dataset)


count = init()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🐱 Cat Facts RAG")
    st.markdown(f"""
    <div class='stat-box'>
        <div style='font-size:2rem'>📚</div>
        <div style='font-size:1.5rem; font-weight:700; color:#7c3aed'>{count}</div>
        <div style='color:#6b7280; font-size:0.85rem'>cat facts loaded</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Models**")
    st.caption("🔍 Embeddings: `bge-base-en-v1.5`")
    st.caption("🤖 LLM: `Llama-3.2-1B-Instruct`")

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by [Ollama](https://ollama.com) + Streamlit")

# --- Main ---
st.markdown("## 🐱 Ask me anything about cats")
st.caption("Answers are grounded in 150 real cat facts.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📎 Sources used"):
                for fact, score in msg["sources"]:
                    st.markdown(f"""
                    <div class='source-card'>
                        <span class='score-badge'>similarity {score}</span><br>{fact}
                    </div>
                    """, unsafe_allow_html=True)

if query := st.chat_input("Ask about cats..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    sources = get_sources(query)

    with st.chat_message("assistant"):
        response = st.write_stream(stream_response(query))
        with st.expander("📎 Sources used"):
            for fact, score in sources:
                st.markdown(f"""
                <div class='source-card'>
                    <span class='score-badge'>similarity {score}</span><br>{fact}
                </div>
                """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })
