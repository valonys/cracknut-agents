import streamlit as st
import requests
import json
import os
import time
from dotenv import load_dotenv
import PyPDF2
import pandas as pd
import openai
from docx import Document
from threading import Thread
from cerebras.cloud.sdk import Cerebras
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
XAI_API_KEY = os.getenv("API_KEY")

st.set_page_config(page_title="DigiTwin RAG", layout="centered")
st.title("📊 DigiTwin Nerdxz")

USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

SYSTEM_PROMPT = (
    "You are DigiTwin, a senior topside inspection engineer. Your task is to analyze and compare the content "
    "of multiple inspection reports, extract trends and KPIs, and provide an evaluation of the last 5 days of operations "
    "along with a predictive progress assessment."
)

# Sidebar
with st.sidebar:
    model_alias = st.selectbox("Choose your AI Agent", [
        "EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys",
        "Llama3 Expert (HF)", "Qwen Inspector (HF)"
    ])
    uploaded_files = st.file_uploader(
        "Upload up to 10 inspection PDFs", type=["pdf"], accept_multiple_files=True
    )

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_contexts" not in st.session_state:
    st.session_state.file_contexts = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None

# Model greeting
if st.session_state.last_model != model_alias:
    if model_alias == "EE Smartest Agent":
        intro = "👋 Hi, I’m **EE**, the Smartest Agent.\n\n- Pragmatic KPI analysis\n- Anomaly detection\n- Document-aware reasoning"
    elif model_alias == "JI Divine Agent":
        intro = "👋 Hi, I’m **JI**, the Divine Agent.\n\n- Predictive logic\n- KPI forecasting\n- Insightful and analytical"
    elif model_alias == "EdJa-Valonys":
        intro = "⚡ Hi, I’m **EdJa-Valonys**.\n\n- Built on LLaMA4\n- Optimized for inspection engineering\n- Fast and structured"
    else:
        intro = None
    if intro:
        st.chat_message("assistant", avatar=BOT_AVATAR).markdown(intro)
        st.session_state.chat_history.append({"role": "assistant", "content": intro})
    st.session_state.last_model = model_alias

# File parsing
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
        return f"Failed to parse file: {e}"

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_hf_model(name):
    model_id = {
        "Llama3 Expert (HF)": "amiguel/Llama3_8B_Instruct_FP16",
        "Qwen Inspector (HF)": "amiguel/GM_Qwen1.8B_Finetune"
    }[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=HF_TOKEN)
    return model, tokenizer

def run_hf_streaming_model(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(target=model.generate, kwargs=dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.7
    ))
    thread.start()
    for new_text in streamer:
        yield new_text

# Process uploaded files
doc_chunks, daily_summaries = [], []

if uploaded_files:
    for file in uploaded_files[:10]:
        text = parse_file(file)
        if text:
            doc_chunks.append(text)
            daily_summaries.append(f"📄 {file.name}:\n{text[:1000]}...\n")

# Build vectorstore if available
if doc_chunks:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents([LCDocument(page_content=t) for t in doc_chunks])
    st.session_state.vectorstore = FAISS.from_documents(split_docs, get_embeddings())

# Response generator
def generate_response(prompt):
    try:
        rag_context = ""
        if daily_summaries:
            rag_context = (
                "\n\n---\nYou are provided with 5-day inspection reports from the same site.\n"
                "Summarize KPI progress trends and predict the next 5-day outlook.\n\n"
                + "\n".join(daily_summaries[:5])
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + rag_context},
            {"role": "user", "content": prompt}
        ]

        if model_alias == "EE Smartest Agent":
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": "grok-beta", "messages": messages, "stream": True},
                stream=True
            )
            for line in response.iter_lines():
                if line:
                    line = line.decode().replace("data: ", "")
                    if line == "[DONE]": break
                    try:
                        data = json.loads(line)
                        if "choices" in data and "delta" in data["choices"][0]:
                            yield data["choices"][0]["delta"].get("content", "")
                    except Exception:
                        continue

        elif model_alias == "JI Divine Agent":
            client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.sambanova.ai/v1")
            stream = client.chat.completions.create(
                model="DeepSeek-R1-Distill-Llama-70B",
                messages=messages,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content.replace("<think>", "").replace("</think>", "")

        elif model_alias == "EdJa-Valonys":
            client = Cerebras(api_key=CEREBRAS_API_KEY)
            result = client.chat.completions.create(
                messages=messages,
                model="llama-4-scout-17b-16e-instruct"
            )
            for word in result.choices[0].message.content.split():
                yield word + " "
                time.sleep(0.02)

        elif model_alias in ["Llama3 Expert (HF)", "Qwen Inspector (HF)"]:
            model, tokenizer = load_hf_model(model_alias)
            full_prompt = SYSTEM_PROMPT + rag_context + "\n\n" + prompt
            for token in run_hf_streaming_model(full_prompt, model, tokenizer):
                yield token

    except Exception as e:
        yield f"⚠️ Error: {e}"

# Show chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

# Chat input and streaming output
if prompt := st.chat_input("Ask about KPIs, anomalies, or forecast..."):
    st.chat_message("user", avatar=USER_AVATAR).markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""
        for token in generate_response(prompt):
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
