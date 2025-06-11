import streamlit as st
import os
import time
import json
import PyPDF2
import pandas as pd
from docx import Document
from threading import Thread
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# Load API keys from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
XAI_API_KEY = os.getenv("API_KEY", None)

# UI
st.set_page_config(page_title="DigiTwin RAG", layout="centered")
st.title("📊 DigiTwin RAG (Local Mode)")

USER_AVATAR = "🧑‍💼"
BOT_AVATAR = "🤖"

SYSTEM_PROMPT = (
    "You are DigiTwin, a senior topside inspection engineer. Analyze the uploaded reports and evaluate "
    "KPI trends, ADHOC tasks, anomalies, and predict a 5-day outlook."
)

# Sidebar
with st.sidebar:
    model_alias = st.selectbox("Choose AI Agent", ["Mock Inspector Agent", "Llama3 Expert (HF)"])
    uploaded_files = st.file_uploader("Upload up to 10 inspection PDFs", type=["pdf"], accept_multiple_files=True)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None

# File parser
def parse_file(file):
    try:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        elif "document" in file.type:
            doc = Document(file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif "excel" in file.type:
            df = pd.read_excel(file)
            return df.to_string()
    except Exception as e:
        return f"Failed to parse: {e}"

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_hf_model():
    model_id = "amiguel/Llama3_8B_Instruct_FP16"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=HF_TOKEN)
    return model, tokenizer

def run_hf_streaming(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(target=model.generate, kwargs=dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.7
    ))
    thread.start()
    for token in streamer:
        yield token

# Vector store with Chroma
doc_chunks, summaries = [], []

if uploaded_files:
    for file in uploaded_files[:10]:
        txt = parse_file(file)
        if txt:
            doc_chunks.append(txt)
            summaries.append(f"📄 {file.name}\n{textwrap.shorten(txt, width=500)}")

    # Build Chroma vector DB
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([LCDocument(page_content=t) for t in doc_chunks])
    st.session_state.vectorstore = Chroma.from_documents(
        docs, embedding=get_embeddings(), persist_directory="chroma_memory"
    )
    st.session_state.vectorstore.persist()

# Welcoming message
if st.session_state.last_model != model_alias:
    intro = f"👋 Hello, I’m **{model_alias}**.\n\n- Ready to summarize inspection KPIs\n- Multi-PDF aware\n- Forecasts site health over 5 days"
    st.chat_message("assistant", avatar=BOT_AVATAR).markdown(intro)
    st.session_state.chat_history.append({"role": "assistant", "content": intro})
    st.session_state.last_model = model_alias

# Response generator
def generate_response(prompt):
    context = ""
    if st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
        context = "\n---\n".join([doc.page_content for doc in docs])

    full_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser: {prompt}"

    if model_alias == "Mock Inspector Agent":
        for word in ("[MOCK REPLY]: Based on your uploaded reports, here's a 5-day KPI outlook...").split():
            yield word + " "
            time.sleep(0.05)

    elif model_alias == "Llama3 Expert (HF)":
        model, tokenizer = load_hf_model()
        for token in run_hf_streaming(full_prompt, model, tokenizer):
            yield token

# Chat UI
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me to summarize or forecast report progress..."):
    st.chat_message("user", avatar=USER_AVATAR).markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        placeholder = st.empty()
        full = ""
        for chunk in generate_response(prompt):
            full += chunk
            placeholder.markdown(full + "▌")
        placeholder.markdown(full)
        st.session_state.chat_history.append({"role": "assistant", "content": full})
