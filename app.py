import streamlit as st
import os
import time
import requests
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
from docx import Document
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from cerebras.cloud.sdk import Cerebras

load_dotenv()

# Font Style
st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/tw-cen-mt');
    * { font-family: 'Tw Cen MT', sans-serif !important; }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="DigiTwin RAG Forecast", layout="centered")
st.title("📊 DigiTwin RAG Forecast App")

# Avatars
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# Bio
ATALIBA_BIO = """
**I am Ataliba Miguel's Digital Twin** 🤖  
- ⛽ 17+ years in Oil & Gas  
- 📋 Expert in Inspection & Maintenance  
- 💼 Founder @ ValonyLabs  
- 💡 Ask me anything about inspection reports or forecast planning!
"""

# Session state init
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_intro_done" not in st.session_state:
    st.session_state.model_intro_done = False
if "current_model" not in st.session_state:
    st.session_state.current_model = None

# Sidebar model + document upload
with st.sidebar:
    model_alias = st.selectbox("Choose your AI Agent", [
        "EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "XAI Inspector", "Valonys Llama"
    ])
    uploaded_files = st.file_uploader("📄 Upload up to 10 PDF reports", type=["pdf"], accept_multiple_files=True)

# Parse PDFs into raw text
def parse_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Build FAISS index
@st.cache_resource
def build_faiss_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for i, doc in enumerate(_docs):
        for chunk in text_splitter.split_text(doc.page_content):
            chunks.append(LCDocument(page_content=chunk, metadata={"source": f"doc_{i}"}))
    return FAISS.from_documents(chunks, embeddings)

# Upload handling
if uploaded_files:
    parsed_docs = [LCDocument(page_content=parse_pdf(f), metadata={"name": f.name}) for f in uploaded_files]
    st.session_state.vectorstore = build_faiss_vectorstore(parsed_docs)
    st.sidebar.success(f"{len(parsed_docs)} reports loaded into memory.")

# System Prompt
SYSTEM_PROMPT = (
    "You are DigiTwin, an expert in inspection & maintenance of offshore systems. "
    "Using uploaded reports, generate summaries, extract KPIs, and forecast progress for the next 5 days. "
    "Be concise, accurate, and technically sound. Use bullet points where needed."
)

# Response generator
def generate_response(prompt):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(prompt, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        messages.append({"role": "system", "content": f"Context from reports:\n{context}"})
    
    messages.append({"role": "user", "content": prompt})
    full_response = ""

    if model_alias == "EE Smartest Agent":
        client = openai.OpenAI(api_key=os.getenv("API_KEY"), base_url="https://api.x.ai/v1")
        response = client.chat.completions.create(
            model="grok-3",
            messages=messages,
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                full_response += delta
                yield f"<span style='font-family:Tw Cen MT'>{delta}</span>"

    elif model_alias == "JI Divine Agent":
        client = openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.sambanova.ai/v1")
        response = client.chat.completions.create(
            model="DeepSeek-R1-Distill-Llama-70B",
            messages=messages,
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"<span style='font-family:Tw Cen MT'>{content}</span>"

    elif model_alias == "EdJa-Valonys":
        client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        response = client.chat.completions.create(model="llama-4-scout-17b-16e-instruct", messages=messages)
        if hasattr(response.choices[0], 'message'):
            content = response.choices[0].message.content
        else:
            content = str(response.choices[0])
        for word in content.split():
            full_response += word + " "
            yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
            time.sleep(0.01)

    elif model_alias == "XAI Inspector":
        model_id = "amiguel/GM_Qwen1.8B_Finetune"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", token=os.getenv("HF_TOKEN"))
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        output = model.generate(input_ids, max_new_tokens=512, do_sample=True, top_p=0.9)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        yield f"<span style='font-family:Tw Cen MT'>{decoded}</span>"

    elif model_alias == "Valonys Llama":
        model_id = "amiguel/Llama3_8B_Instruct_FP16"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=os.getenv("HF_TOKEN"))
        input_ids = tokenizer(SYSTEM_PROMPT + "\n\n" + prompt, return_tensors="pt").to(model.device)
        output = model.generate(**input_ids, max_new_tokens=512)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        yield f"<span style='font-family:Tw Cen MT'>{decoded}</span>"

# Welcoming Message
if not st.session_state.model_intro_done or st.session_state.current_model != model_alias:
    if model_alias == "EE Smartest Agent":
        intro = "**EE Agent Activated** — Pragmatic, Innovative, Smart 💡"
    elif model_alias == "JI Divine Agent":
        intro = "**JI Agent Activated** — Gifted with divine LLM powers ✨"
    elif model_alias == "EdJa-Valonys":
        intro = "**EdJa Agent Activated** — Cerebras-fast ⚡"
    elif model_alias == "XAI Inspector":
        intro = "**XAI Inspector Activated** — Custom-trained Qwen on inspections 🔍"
    elif model_alias == "Valonys Llama":
        intro = "**Valonys Llama Activated** — LLaMA3-based inspection expert 🦙"

    st.session_state.chat_history.append({"role": "assistant", "content": intro})
    st.session_state.model_intro_done = True
    st.session_state.current_model = model_alias

# Chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Prompt input
if prompt := st.chat_input("Ask a summary or forecast about the reports..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in generate_response(prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
        response_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
