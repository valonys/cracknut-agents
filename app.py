import streamlit as st
import requests
import json
import os
import time
from dotenv import load_dotenv
import PyPDF2
from docx import Document
import pandas as pd
import openai  # For DeepSeek API
from cerebras.cloud.sdk import Cerebras  # For Cerebras API

# --- RAG Imports ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
import tempfile

# --- ENV ---
load_dotenv()

# --- Avatars ---
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# --- System Prompt ---
SYSTEM_PROMPT = (
    "You are DigiTwin, a digital expert and senior topside engineer specializing in inspection and maintenance "
    "of offshore piping systems, structural elements, mechanical equipment, floating production units, pressure vessels "
    "(with emphasis on Visual Internal Inspection - VII), and pressure safety devices (PSDs). Rely on uploaded documents "
    "and context to provide practical, standards-driven, and technically accurate responses. Your guidance reflects deep "
    "field experience, industry regulations, and proven methodologies in asset integrity and reliability engineering."
)

# --- App Style ---
st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/tw-cen-mt');
    * { font-family: 'Tw Cen MT', sans-serif; }
    .st-emotion-cache-1y4p8pa { padding: 2rem 1rem; }
    </style>
""", unsafe_allow_html=True)
st.title("🚀 Ataliba o Agent Nerdx 🚀")

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    model_alias = st.selectbox(
        "Choose your AI Agent",
        options=["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys"],
        index=0,
        help="Select the AI model for your session."
    )

    st.header("📁 Document Hub")
    uploaded_files = st.file_uploader(
        "Upload technical documents (PDF only)", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        st.session_state.faiss_db = None  # Reset on new upload

# --- FAISS Memory Setup ---
@st.cache_resource
def build_faiss_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(_docs, embeddings)

def process_uploaded_pdfs(files):
    all_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp.flush()
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return build_faiss_vectorstore(splitter.split_documents(all_docs))

def retrieve_context(query, db, k=4):
    retrieved_docs = db.similarity_search(query, k=k)
    return "\n---\n".join(doc.page_content for doc in retrieved_docs)

# --- Session Setup ---
if "file_context" not in st.session_state:
    st.session_state.file_context = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_intro_done" not in st.session_state:
    st.session_state.model_intro_done = False
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if uploaded_files and "faiss_db" not in st.session_state:
    st.session_state.faiss_db = process_uploaded_pdfs(uploaded_files)

# --- Model Introductions ---
if not st.session_state.model_intro_done or st.session_state.current_model != model_alias:
    if model_alias == "EE Smartest Agent":
        intro_message = """
        Hi, I am **EE**, the Double E Agent! 🚀  
        - **Pragmatic**: I solve problems efficiently  
        - **Innovative**: My reasoning goes beyond human limits  
        - **Smart**: I outclass most systems  
        """
    elif model_alias == "JI Divine Agent":
        intro_message = """
        Hi, I am **JI**, the Divine Agent! ✨  
        - **Gifted**: Advanced reasoning  
        - **Quasi-Human**: Intuitive intelligence  
        - **Divine**: Unmatched capabilities  
        """
    elif model_alias == "EdJa-Valonys":
        intro_message = """
        Greetings, I am **EdJa-Valonys**! ⚡  
        - **Lightning-fast** Cerebras inference  
        - **Llama-4 core**  
        - **Industrial-grade** excellence  
        """
    st.session_state.chat_history.append({"role": "assistant", "content": intro_message})
    st.session_state.model_intro_done = True
    st.session_state.current_model = model_alias

# --- Display Chat History ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

# --- Chat Input Handler ---
if prompt := st.chat_input("Ask about documents or technical matters..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""

        # Inject RAG context if available
        if "faiss_db" in st.session_state:
            context = retrieve_context(prompt, st.session_state.faiss_db)
            final_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {prompt}"
        else:
            final_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {prompt}"

        # Dummy response generator – replace with actual model call
        def generate_response(query):
            yield "🔧 This is a placeholder response. Connect your model inference here."

        for chunk in generate_response(final_prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
