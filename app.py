import streamlit as st
import os
import time
import json
import requests
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
import openai
import tempfile

# LangChain v0.2+ compatible imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter

# --- ENV ---
load_dotenv()

# --- UI Setup ---
st.set_page_config(page_title="DigiTwin RAG", layout="wide")
st.title("🚀 Ataliba o Agent Nerdx 🚀")

USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

SYSTEM_PROMPT = (
    "You are DigiTwin, a senior topside engineer and inspection specialist. Your responses must be technically accurate, "
    "aligned with oil & gas standards, and informed by uploaded document context including piping, structures, vessels, "
    "PSDs, floating units, and topside inspection strategies."
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    model_alias = st.selectbox("Choose your AI Agent", ["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys"])

    st.header("📁 Upload PDF Docs")
    uploaded_files = st.file_uploader("Upload up to 10 PDFs", type=["pdf"], accept_multiple_files=True)

# --- State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "faiss_db" not in st.session_state:
    st.session_state.faiss_db = None
if "faiss_initialized" not in st.session_state:
    st.session_state.faiss_initialized = False

# --- PDF Processing ---
@st.cache_resource
def get_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def parse_pdfs(files):
    all_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp.flush()
            loader = PyPDFLoader(tmp.name)
            pages = loader.load()
            all_docs.extend(pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(all_docs)

# --- On File Upload ---
if uploaded_files:
    if not st.session_state.faiss_initialized:
        try:
            split_docs = parse_pdfs(uploaded_files)
            st.session_state.faiss_db = get_vectorstore(split_docs)
            st.session_state.faiss_initialized = True
            st.sidebar.success(f"✅ {len(split_docs)} document chunks indexed.")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to process PDFs: {e}")

# --- Retrieve Context ---
def retrieve_context(query, db, k=4):
    try:
        results = db.similarity_search(query, k=k)
        return "\n---\n".join([doc.page_content for doc in results])
    except Exception as e:
        st.warning(f"⚠️ Document retrieval failed: {e}")
        return ""

# --- Chat Display ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

# --- Prompt Input ---
if prompt := st.chat_input("Ask about inspection, maintenance, or documents..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""

        # Context injection
        context = retrieve_context(prompt, st.session_state.faiss_db) if st.session_state.faiss_db else ""
        final_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {prompt}"

        def generate_response(prompt):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            try:
                if model_alias == "EE Smartest Agent":
                    response = requests.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {os.getenv('API_KEY')}", "Content-Type": "application/json"},
                        json={"model": "grok-beta", "messages": messages, "temperature": 0.2, "stream": True},
                        stream=True
                    )
                    for line in response.iter_lines():
                        if line:
                            chunk = line.decode("utf-8").replace("data: ", "")
                            if chunk == "[DONE]": break
                            data = json.loads(chunk)
                            delta = data["choices"][0]["delta"].get("content", "")
                            yield delta

                elif model_alias == "JI Divine Agent":
                    client = openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.sambanova.ai/v1")
                    response = client.chat.completions.create(
                        model="DeepSeek-R1-Distill-Llama-70B",
                        messages=messages,
                        temperature=0.1,
                        top_p=0.1,
                        stream=True
                    )
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

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

        for chunk in generate_response(final_prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
