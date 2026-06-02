import streamlit as st
from main import load_dataset, build_vector_db, stream_response

st.title("🐱 Cat Facts Chatbot")

@st.cache_resource(show_spinner="Loading dataset...")
def init():
    dataset = load_dataset()
    build_vector_db(dataset)
    return len(dataset)

count = init()
st.caption(f"Loaded {count} cat facts into vector DB.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Ask me about cats..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        response = st.write_stream(stream_response(query))

    st.session_state.messages.append({"role": "assistant", "content": response})
