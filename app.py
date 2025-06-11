import streamlit as st
import os
import time
import json
from dotenv import load_dotenv
import PyPDF2
from docx import Document
import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
import openai  # For DeepSeek API
from cerebras.cloud.sdk import Cerebras

# Load secrets and .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

# App config
st.set_page_config(page_title="DigiTwin RAG", page_icon="📂", layout="centered")
st.title("🚀 Ataliba o Agent Nerdx 🚀")

# Sidebar selection
with st.sidebar:
    model_alias = st.selectbox(
        "Choose your AI Agent",
        options=[
            "EE Smartest Agent (Grok)", 
            "JI Divine Agent (DeepSeek)", 
            "EdJa-Valonys (Cerebras)", 
            "DigiTwin Agent (HF RAG)"
        ],
        index=0
    )
    uploaded_file = st.file_uploader("Upload technical documents", type=["pdf", "docx", "xlsx", "xlsm"])

# System Prompt
SYSTEM_PROMPT = (
    "You are DigiTwin, a digital expert and senior topside engineer specializing in inspection and maintenance "
    "of offshore piping systems, structural elements, mechanical equipment, floating production units, pressure vessels "
    "(with emphasis on Visual Internal Inspection - VII), and pressure safety devices (PSDs). Rely on uploaded documents "
    "and context to provide practical, standards-driven, and technically accurate responses. Your guidance reflects deep "
    "field experience, industry regulations, and proven methodologies in asset integrity and reliability engineering."
)

@st.cache_resource(show_spinner="Loading embeddings...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Creating vector index...")
def build_vectorstore(raw_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([LCDocument(page_content=d) for d in raw_docs])
    return FAISS.from_documents(docs, get_embeddings())

@st.cache_resource(show_spinner="Loading HF model...")
def load_model():
    model_id = "amiguel/Llama3_8B_Instruct_FP16"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=HF_TOKEN)
    return model, tokenizer

def parse_file(file):
    try:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            df = pd.read_excel(file)
            return df.to_string()
    except Exception as e:
        st.error(f"Error processing file: {e}")
    return ""

def run_rag(prompt, retriever, model, tokenizer):
    docs = retriever.get_relevant_documents(prompt)
    context = "\n---\n".join([d.page_content for d in docs])
    full_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser Query: {prompt}\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load and index document if applicable
if uploaded_file and model_alias == "DigiTwin Agent (HF RAG)":
    raw_text = parse_file(uploaded_file)
    if raw_text:
        st.session_state.vectorstore = build_vectorstore([raw_text])

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your inspection/maintenance question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if model_alias == "DigiTwin Agent (HF RAG)" and "vectorstore" in st.session_state:
            model, tokenizer = load_model()
            rag_output = run_rag(prompt, st.session_state.vectorstore.as_retriever(), model, tokenizer)
            st.markdown(rag_output)
            st.session_state.chat_history.append({"role": "assistant", "content": rag_output})
        elif model_alias.startswith("EE"):
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {XAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-beta",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                }
            )
            reply = response.json()["choices"][0]["message"]["content"]
            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        elif model_alias.startswith("JI"):
            client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.sambanova.ai/v1")
            completion = client.chat.completions.create(
                model="DeepSeek-R1-Distill-Llama-70B",
                messages=[{"role": "user", "content": prompt}]
            )
            reply = completion.choices[0].message.content
            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        elif model_alias.startswith("EdJa"):
            client = Cerebras(api_key=CEREBRAS_API_KEY)
            result = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-4-scout-17b-16e-instruct"
            )
            reply = result.choices[0].message.content
            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        else:
            st.markdown("⚠️ Unsupported model selection.")
