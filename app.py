import streamlit as st
import os
import time
import requests
import json
import PyPDF2
import pandas as pd
from docx import Document as DocxDocument
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from langchain.vectorstores import FAISS
from langchain.schema import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

st.set_page_config(page_title="DigiTwin Forecast", page_icon="📊")
st.title("📊 DigiTwin RAG Forecast")

with st.sidebar:
    model_alias = st.selectbox("Choose Model", ["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "HF-Llama3", "HF-Qwen"], index=0)
    uploaded_files = st.file_uploader("Upload up to 10 PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

for key in ["chat_history", "model_intro_done", "current_model", "vectorstore"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "vectorstore" else []

SYSTEM_PROMPT = (
    "You are DigiTwin, a digital expert and senior topside engineer specializing in inspection and maintenance. "
    "Use the uploaded reports to summarize inspection KPIs, trends, and forecast the next 5-day progress."
)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def parse_file(file):
    try:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() for page in reader.pages])
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = DocxDocument(file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Failed to parse {file.name}: {str(e)}")
        return None

def build_faiss_vectorstore(_docs):
    return FAISS.from_documents(_docs, embedding=get_embeddings())

if uploaded_files:
    raw_texts = []
    for file in uploaded_files[:10]:
        text = parse_file(file)
        if text:
            raw_texts.append(text)

    if raw_texts:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = splitter.split_documents([LCDocument(page_content=txt) for txt in raw_texts])
        st.session_state.vectorstore = build_faiss_vectorstore(documents)
        st.sidebar.success("✅ Reports indexed successfully")

def search_context(query):
    if st.session_state.vectorstore:
        results = st.session_state.vectorstore.similarity_search(query, k=4)
        return "\n\n".join([doc.page_content for doc in results])
    return ""

def model_intro():
    if model_alias == "EE Smartest Agent":
        return "**EE Agent**: Pragmatic, Innovative, and Smart. Specialized in reasoning technical KPIs."
    elif model_alias == "JI Divine Agent":
        return "**JI Agent**: Gifted and quasi-human. Trained in inspection reasoning."
    elif model_alias == "EdJa-Valonys":
        return "**EdJa-Valonys**: Cerebras-powered lightning-fast precision agent."
    elif model_alias == "HF-Llama3":
        return "**HF-Llama3**: Locally fine-tuned LLaMA3 inspection model."
    elif model_alias == "HF-Qwen":
        return "**HF-Qwen**: Customized Qwen model trained on maintenance reports."

def generate_response(prompt):
    context = search_context(prompt)
    system = SYSTEM_PROMPT + "\n\nContext:\n" + context if context else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    full_response = ""

    try:
        if model_alias == "EE Smartest Agent":
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('API_KEY')}", "Content-Type": "application/json"},
                json={"model": "grok-beta", "messages": messages, "temperature": 0.3, "stream": True},
                stream=True
            )
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode().replace("data: ", ""))
                    delta = chunk['choices'][0]['delta'].get('content', '')
                    full_response += delta
                    yield delta

        elif model_alias == "JI Divine Agent":
            import openai
            client = openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.sambanova.ai/v1")
            response = client.chat.completions.create(model="DeepSeek-R1-Distill-Llama-70B", messages=messages, stream=True)
            for chunk in response:
                delta = chunk.choices[0].delta.content
                full_response += delta
                yield delta

        elif model_alias == "EdJa-Valonys":
            from cerebras.cloud.sdk import Cerebras
            client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
            resp = client.chat.completions.create(model="llama-4-scout-17b-16e-instruct", messages=messages)
            full_response = resp.choices[0].message.content
            for word in full_response.split():
                yield word + " "
                time.sleep(0.05)

        elif model_alias in ["HF-Llama3", "HF-Qwen"]:
            model_id = "amiguel/Llama3_8B_Instruct_FP16" if model_alias == "HF-Llama3" else "amiguel/GM_Qwen1.8B_Finetune"
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True, token=HF_TOKEN)
            input_ids = tokenizer.encode(system + "\n\n" + prompt, return_tensors="pt").to(model.device)
            streamer = TextIteratorStreamer(tokenizer)
            gen_thread = Thread(target=model.generate, kwargs={"inputs": input_ids, "max_new_tokens": 512, "streamer": streamer})
            gen_thread.start()
            for token in streamer:
                yield token

    except Exception as e:
        yield f"⚠️ Error: {str(e)}"

# Welcome message
if not st.session_state.model_intro_done or st.session_state.current_model != model_alias:
    intro = model_intro()
    if intro:
        st.session_state.chat_history.append({"role": "assistant", "content": intro})
        st.session_state.model_intro_done = True
        st.session_state.current_model = model_alias

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about reports or progress forecast..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in generate_response(prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
