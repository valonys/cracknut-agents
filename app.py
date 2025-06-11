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
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

ATALIBA_BIO = """
**I am Ataliba Miguel's Digital Twin** 🤖

**Background:**
- 🎓 Mechanical Engineering (BSc)
- ⛽ Oil & Gas Engineering (MSc Specialization)
- 🔧 17+ years in Oil & Gas Industry
- 🔍 Topside Inspection Methods Engineer @ TotalEnergies
- 🤖 AI Practitioner Specialist
- 🚀 Founder of ValonyLabs
"""

SYSTEM_PROMPT = (
    "You are DigiTwin, a digital expert in inspection and maintenance for offshore facilities, piping systems, "
    "mechanical equipment, pressure vessels (Visual Internal Inspection - VII), and pressure safety devices (PSDs). "
    "Use uploaded document context to provide clear, technical, and standards-based answers."
)

st.set_page_config(page_title="DigiTwin RAG", layout="centered")
st.title("🚀 Ataliba o Agent Nerdx 🚀")

with st.sidebar:
    st.header("AI Agent")
    model_alias = st.selectbox("Choose a model", [
        "EE Smartest Agent", 
        "JI Divine Agent", 
        "EdJa-Valonys", 
        "Llama3 Expert (HF)", 
        "Qwen Inspector (HF)"
    ])
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "xlsx", "xlsm"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

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
    except Exception as err:
        return f"Error reading file: {err}"
    return ""

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

def run_rag(prompt, retriever, model, tokenizer):
    docs = retriever.get_relevant_documents(prompt)
    context = "\n---\n".join([d.page_content for d in docs])
    full_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser Query: {prompt}\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    output = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# File upload and memory
if uploaded_file:
    file_text = parse_file(uploaded_file)
    st.session_state.vectorstore = build_vectorstore(file_text)
    st.sidebar.success("✅ Document processed and embedded")

# Chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

# Chat input and model execution
if prompt := st.chat_input("Ask me about inspection, documents or maintenance..."):
    st.chat_message("user", avatar=USER_AVATAR).markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        if any(t in prompt.lower() for t in ["ataliba", "yourself", "valonylabs", "who are you"]):
            st.markdown(ATALIBA_BIO)
            st.session_state.chat_history.append({"role": "assistant", "content": ATALIBA_BIO})
        elif model_alias in ["Llama3 Expert (HF)", "Qwen Inspector (HF)"]:
            model, tokenizer = load_hf_model(model_alias)
            retriever = st.session_state.vectorstore.as_retriever() if st.session_state.vectorstore else build_vectorstore("").as_retriever()
            response = run_rag(prompt, retriever, model, tokenizer)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        elif model_alias == "EE Smartest Agent":
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": "grok-beta", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
            )
            reply = response.json()["choices"][0]["message"]["content"]
            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        elif model_alias == "JI Divine Agent":
            client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.sambanova.ai/v1")
            resp = client.chat.completions.create(model="DeepSeek-R1-Distill-Llama-70B", messages=[{"role": "user", "content": prompt}])
            reply = resp.choices[0].message.content
            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        elif model_alias == "EdJa-Valonys":
            client = Cerebras(api_key=CEREBRAS_API_KEY)
            resp = client.chat.completions.create(model="llama-4-scout-17b-16e-instruct", messages=[{"role": "user", "content": prompt}])
            reply = resp.choices[0].message.content
            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        else:
            st.markdown("⚠️ Unsupported model.")
