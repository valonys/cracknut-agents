import streamlit as st
import os
import time
import json
import PyPDF2
import openai
import requests
import pandas as pd
from docx import Document
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument

# Load keys
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
XAI_KEY = os.getenv("API_KEY", "")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
CEREBRAS_KEY = os.getenv("CEREBRAS_API_KEY", "")

# UI config
st.set_page_config(page_title="DigiTwin RAG", layout="centered")
st.title("📊 DigiTwin RAG Forecast")

USER_AVATAR = "🧑‍💼"
BOT_AVATAR = "🤖"

SYSTEM_PROMPT = (
    "You are DigiTwin, a senior offshore topside inspection engineer. "
    "Your job is to analyze daily inspection reports uploaded, summarize the KPIs (especially Compl, Not Compl), "
    "identify any backlog trends or issues, and forecast likely work execution trends over the next 5 days."
)

# Sidebar
with st.sidebar:
    model_alias = st.selectbox("Choose AI Agent", ["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "Llama3 Expert"])
    uploaded_files = st.file_uploader("📁 Upload up to 10 PDF reports", type=["pdf"], accept_multiple_files=True)

# State init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_model" not in st.session_state:
    st.session_state.last_model = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# File parsing
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

# FAISS vectorstore build
@st.cache_resource
def build_faiss_vectorstore(docs):
    return FAISS.from_documents(docs, embedding=get_embeddings())

#@st.cache_resource
#def build_faiss_vectorstore(_docs):
#    return FAISS.from_documents(_docs, embedding=get_embeddings())



# Build doc memory
doc_chunks = []
if uploaded_files:
    for file in uploaded_files[:10]:
        content = parse_file(file)
        if content:
            doc_chunks.append(content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([LCDocument(page_content=txt) for txt in doc_chunks])
    st.session_state.vectorstore = build_faiss_vectorstore(docs)

# Welcome messages
if st.session_state.last_model != model_alias:
    welcome = {
        "EE Smartest Agent": "Hi, I'm **EE**, the Double E Agent 🚀\n- Efficient • Expert • Engineered for Accuracy",
        "JI Divine Agent": "Greetings, I'm **JI**, the Divine Agent ✨\n- Gifted Reasoning • Quasi-Human Logic",
        "EdJa-Valonys": "Hello, I'm **EdJa-Valonys** ⚡\n- Cerebras-speed • Built for Engineering Logic",
        "Llama3 Expert": "Hi, I'm **Llama3 Expert** 🦙\n- Finetuned for industrial inspection insight"
    }
    st.chat_message("assistant", avatar=BOT_AVATAR).markdown(welcome[model_alias])
    st.session_state.chat_history.append({"role": "assistant", "content": welcome[model_alias]})
    st.session_state.last_model = model_alias

# Generate response
def generate_response(prompt):
    context = ""
    if st.session_state.vectorstore:
        retrieved = st.session_state.vectorstore.similarity_search(prompt, k=4)
        context = "\n\n".join([doc.page_content for doc in retrieved])

    full_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser: {prompt}"

    if model_alias == "EE Smartest Agent":
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {XAI_KEY}", "Content-Type": "application/json"},
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
        for token in run_hf_streaming(full_prompt, model, tokenizer):
            yield token

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

# Prompt and Response UI
if prompt := st.chat_input("Ask a forecast, KPI summary, or anomaly report..."):
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
