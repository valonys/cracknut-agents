import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
import time
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
import openai
import requests
import json

# FAISS + Embeddings
from sentence_transformers import SentenceTransformer
import faiss

# --- Load env
load_dotenv()

# --- UI Setup
st.set_page_config(page_title="DigiTwin RAG", layout="wide")
st.markdown("<h1 style='font-family:Tw Cen MT;'>🚀 Ataliba o Agent Nerdx 🚀</h1>", unsafe_allow_html=True)

USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

SYSTEM_PROMPT = (
    "You are DigiTwin, a senior topside engineer and inspection specialist. Your responses must be technically accurate, "
    "aligned with oil & gas standards, and informed by uploaded document context including piping, structures, vessels, "
    "PSDs, floating units, and topside inspection strategies."
)

# --- Sidebar
with st.sidebar:
    st.header("⚙️ Model Selection")
    model_alias = st.selectbox("Choose your AI Agent", ["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys"])

    st.header("📁 Upload PDF Docs")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# --- Memory
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = get_embedding_model()

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_top_k(query, chunks, embeddings, index, k=4):
    query_vec = embed_model.encode([query])
    D, I = index.search(query_vec, k)
    return "\n---\n".join([chunks[i] for i in I[0]])

# --- On upload
if uploaded_files:
    new_chunks = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        split = [text[i:i+800] for i in range(0, len(text), 800)]
        new_chunks.extend(split)

    st.session_state.doc_chunks = new_chunks
    st.session_state.vector_index, st.session_state.embeddings_matrix = build_faiss_index(new_chunks)
    st.sidebar.success(f"✅ {len(uploaded_files)} PDFs processed with {len(new_chunks)} chunks.")

# --- Display chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

# --- User prompt
if prompt := st.chat_input("Ask anything based on the docs..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""

        # RAG Context
        try:
            context = retrieve_top_k(prompt, st.session_state.doc_chunks, st.session_state.embeddings_matrix, st.session_state.vector_index)
            final_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{prompt}"
        except Exception as e:
            final_prompt = f"{SYSTEM_PROMPT}\n\nQuestion:\n{prompt}"
            st.warning(f"⚠️ Document retrieval failed: {e}")

        # Model Routing
        def generate_response(prompt):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            try:
                if model_alias == "EE Smartest Agent":
                    response = requests.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {os.getenv('API_KEY')}", "Content-Type": "application/json"},
                        json={"model": "grok-beta", "messages": messages, "temperature": 0.2, "stream": True},
                        stream=True,
                    )
                    for line in response.iter_lines():
                        if line:
                            chunk = line.decode("utf-8").replace("data: ", "")
                            if chunk == "[DONE]": break
                            data = json.loads(chunk)
                            delta = data["choices"][0]["delta"].get("content", "")
                            yield delta

                elif model_alias == "JI Divine Agent":
                    client = openai.OpenAI(
                        api_key=os.getenv("DEEPSEEK_API_KEY"),
                        base_url="https://api.sambanova.ai/v1"
                    )
                    response = client.chat.completions.create(
                        model="DeepSeek-R1-Distill-Llama-70B",
                        messages=messages,
                        temperature=0.1,
                        top_p=0.1,
                        stream=True
                    )
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            yield content

                elif model_alias == "EdJa-Valonys":
                    client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
                    response = client.chat.completions.create(
                        model="llama-4-scout-17b-16e-instruct",
                        messages=messages,
                    )
                    full = response.choices[0].message.content
                    for word in full.split():
                        yield word + " "
                        time.sleep(0.03)
                    yield ""

            except Exception as e:
                yield f"⚠️ API Error: {e}"

        # Stream output
        for chunk in generate_response(final_prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
