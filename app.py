import streamlit as st
import requests
import json
import os
import time
from dotenv import load_dotenv
import PyPDF2
from docx import Document
import pandas as pd
import openai
from cerebras.cloud.sdk import Cerebras
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument

# --- Load secrets and keys ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
XAI_API_KEY = os.getenv("API_KEY")

# --- UI settings ---
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

SYSTEM_PROMPT = (
    "You are DigiTwin, a digital expert in inspection and maintenance for offshore facilities, "
    "piping systems, pressure vessels (VII), PSDs, and topside mechanical assets. Your answers are "
    "precise, actionable, and based on both domain knowledge and the context provided from reports or documents."
)

st.set_page_config(page_title="DigiTwin RAG", layout="centered")
st.title("🚀 Ataliba o Agent Nerdx 🚀")

# Sidebar
with st.sidebar:
    model_alias = st.selectbox("Choose your AI Agent", [
        "EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys",
        "Llama3 Expert (HF)", "Qwen Inspector (HF)"
    ])
    uploaded_file = st.file_uploader("Upload inspection report", type=["pdf", "docx", "xlsx", "xlsm"])

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "file_context" not in st.session_state:
    st.session_state.file_context = ""

# File processing
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

# Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_hf_model(name):
    models = {
        "Llama3 Expert (HF)": "amiguel/Llama3_8B_Instruct_FP16",
        "Qwen Inspector (HF)": "amiguel/GM_Qwen1.8B_Finetune"
    }
    model_id = models[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=HF_TOKEN)
    return model, tokenizer

def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([LCDocument(page_content=text)])
    return FAISS.from_documents(docs, get_embeddings())

# Process file and embed
if uploaded_file:
    st.session_state.file_context = parse_file(uploaded_file)
    if st.session_state.file_context:
        st.session_state.vectorstore = build_vectorstore(st.session_state.file_context)
        st.sidebar.success("✅ Document parsed and embedded.")

# Generate token-by-token response
def generate_response(prompt):
    try:
        doc_summary = ""
        if st.session_state.file_context:
            doc_summary = (
                "\n\n---\nSUMMARY OF INSPECTION REPORT:\n"
                "The attached report contains inspection data including VIE, VII, UTM measurements, "
                "corrosion anomalies, PSV tests, WBT access notes, and pending SAP activities. "
                "Follow-up campaigns and next-day inspection tasks are outlined.\n\n"
                f"RAW CONTEXT:\n{st.session_state.file_context[:2000]}\n"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + doc_summary},
            {"role": "user", "content": prompt}
        ]

        # EE Agent
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
                    yield json.loads(line)["choices"][0]["delta"].get("content", "")

        # DeepSeek (JI)
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

        # Cerebras
        elif model_alias == "EdJa-Valonys":
            client = Cerebras(api_key=CEREBRAS_API_KEY)
            out = client.chat.completions.create(
                messages=messages,
                model="llama-4-scout-17b-16e-instruct"
            )
            txt = out.choices[0].message.content
            for token in txt.split():
                yield token + " "
                time.sleep(0.03)

    except Exception as e:
        yield f"⚠️ Error: {e}"

# Chat history display
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something technical or document-related..."):
    st.chat_message("user", avatar=USER_AVATAR).markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in generate_response(prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
