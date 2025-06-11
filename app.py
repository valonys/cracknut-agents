import streamlit as st
import os
import time
import json
import PyPDF2
import textwrap
import pandas as pd
from docx import Document
from threading import Thread
from dotenv import load_dotenv

import openai
import requests
from cerebras.cloud.sdk import Cerebras
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument

# Load .env keys
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
XAI_KEY = os.getenv("API_KEY", None)
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", None)
CEREBRAS_KEY = os.getenv("CEREBRAS_API_KEY", None)

# UI Setup
st.set_page_config(page_title="DigiTwin RAG", layout="centered")
st.title("📊 DigiTwin RAG Forecast")

USER_AVATAR = "🧑‍💼"
BOT_AVATAR = "🤖"

SYSTEM_PROMPT = (
    "You are DigiTwin, a senior offshore topside inspection engineer. Analyze uploaded inspection reports "
    "to extract key KPIs, ADHOC maintenance trends, anomalies, and forecast progress outlook over the next 5 days."
)

# Sidebar
with st.sidebar:
    model_alias = st.selectbox("Choose AI Agent", ["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "Llama3 Expert"])
    uploaded_files = st.file_uploader("📁 Upload up to 10 PDF reports", type=["pdf"], accept_multiple_files=True)

# State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_model" not in st.session_state:
    st.session_state.last_model = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Parse uploaded files
def parse_file(file):
    try:
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        return f"Could not parse {file.name}: {e}"

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

# Process file upload and build vectorstore
doc_chunks = []
if uploaded_files:
    for file in uploaded_files[:10]:
        content = parse_file(file)
        if content:
            doc_chunks.append(content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([LCDocument(page_content=txt) for txt in doc_chunks])
    st.session_state.vectorstore = Chroma.from_documents(docs, get_embeddings(), persist_directory="chroma_memory")
    st.session_state.vectorstore.persist()

# Welcoming messages
if st.session_state.last_model != model_alias:
    if model_alias == "EE Smartest Agent":
        msg = """Hi, I'm **EE**, the Double E Agent 🚀  
- Efficient • Expert • Engineered for accuracy"""
    elif model_alias == "JI Divine Agent":
        msg = """Greetings, I'm **JI**, the Divine Agent ✨  
- Gifted in reasoning • Quasi-human logic • Deep insight"""
    elif model_alias == "EdJa-Valonys":
        msg = """Hello, I'm **EdJa-Valonys** ⚡  
- Cerebras-speed • Built for industry • Answers grounded in engineering logic"""
    elif model_alias == "Llama3 Expert":
        msg = """Hi, I'm **Llama3 Expert** 🦙  
- Powered by Hugging Face  
- Specialized for inspection and maintenance insights"""
    st.chat_message("assistant", avatar=BOT_AVATAR).markdown(msg)
    st.session_state.chat_history.append({"role": "assistant", "content": msg})
    st.session_state.last_model = model_alias

# Generate response
def generate_response(prompt):
    context = ""
    if st.session_state.vectorstore:
        top_docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
        context = "\n\n".join([doc.page_content for doc in top_docs])

    final_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser: {prompt}"

    if model_alias == "EE Smartest Agent":
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {XAI_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-beta",
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                "stream": True
            },
            stream=True
        )
        for line in response.iter_lines():
            if line:
                chunk = line.decode('utf-8').replace("data: ", "")
                if chunk == "[DONE]": break
                try:
                    data = json.loads(chunk)
                    if "choices" in data and "delta" in data["choices"][0]:
                        delta = data["choices"][0]["delta"].get("content", "")
                        yield delta
                except:
                    continue

    elif model_alias == "JI Divine Agent":
        client = openai.OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.sambanova.ai/v1")
        stream = client.chat.completions.create(
            model="DeepSeek-R1-Distill-Llama-70B",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content
                yield content

    elif model_alias == "EdJa-Valonys":
        client = Cerebras(api_key=CEREBRAS_KEY)
        result = client.chat.completions.create(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            model="llama-4-scout-17b-16e-instruct"
        )
        text = result.choices[0].message.content
        for word in text.split():
            yield word + " "
            time.sleep(0.02)

    elif model_alias == "Llama3 Expert":
        model, tokenizer = load_hf_model()
        for token in run_hf_streaming(final_prompt, model, tokenizer):
            yield token

# Chat history UI
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask a forecast or summary..."):
    st.chat_message("user", avatar=USER_AVATAR).markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full = ""
        for chunk in generate_response(prompt):
            full += chunk
            response_placeholder.markdown(full + "▌")
        response_placeholder.markdown(full)
        st.session_state.chat_history.append({"role": "assistant", "content": full})
