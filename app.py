import os
import time
import json
import streamlit as st
from dotenv import load_dotenv
from threading import Thread
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import openai
import requests

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Streamlit app config and font
st.set_page_config(page_title="Forecast RAG App", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/tw-cen-mt');
    * { font-family: 'Tw Cen MT', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# UI Avatars
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# Prompt
SYSTEM_PROMPT = (
    "You are DigiTwin, a senior topside inspection and maintenance engineer. "
    "Use uploaded inspection reports, KPIs, and industry best practices to evaluate progress, identify trends, and forecast work."
)

# Sidebar
with st.sidebar:
    uploaded_files = st.file_uploader("📁 Upload up to 10 PDFs", type=["pdf"], accept_multiple_files=True)
    model_alias = st.selectbox("Choose Agent", ["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys"])

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False
if "current_model" not in st.session_state:
    st.session_state.current_model = None

# PDF parsing
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def chunk_and_embed_text(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = splitter.split_documents([Document(page_content=d) for d in docs])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)

if uploaded_files and not st.session_state.vectorstore:
    contents = [extract_text_from_pdf(file) for file in uploaded_files]
    st.session_state.vectorstore = chunk_and_embed_text(contents)
    st.sidebar.success("✅ Documents processed successfully")

# RAG context
def search_context(query):
    if st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(query, k=4)
        return "\n\n".join([doc.page_content for doc in docs])
    return ""

# Main response generation
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
                    try:
                        chunk = json.loads(line.decode().replace("data: ", ""))
                        if "choices" in chunk and "delta" in chunk["choices"][0]:
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            full_response += delta
                            yield f"<span style='font-family:Tw Cen MT'>{delta}</span>"
                    except Exception as e:
                        yield f"<span style='font-family:Tw Cen MT'>⚠️ API Parse Error: {str(e)}</span>"

        elif model_alias == "JI Divine Agent":
            client = openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.sambanova.ai/v1")
            response = client.chat.completions.create(model="DeepSeek-R1-Distill-Llama-70B", messages=messages, stream=True)
            for chunk in response:
                delta = chunk.choices[0].delta.content
                full_response += delta
                yield f"<span style='font-family:Tw Cen MT'>{delta}</span>"

        elif model_alias == "EdJa-Valonys":
            from cerebras.cloud.sdk import Cerebras
            client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
            result = client.chat.completions.create(model="llama-4-scout-17b-16e-instruct", messages=messages)
            full_response = result.choices[0].message.content
            for word in full_response.split():
                yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                time.sleep(0.05)

        elif model_alias in ["HF-Llama3", "HF-Qwen"]:
            model_id = "amiguel/Llama3_8B_Instruct_FP16" if model_alias == "HF-Llama3" else "amiguel/GM_Qwen1.8B_Finetune"
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto", token=HF_TOKEN)
            input_ids = tokenizer.encode(system + "\n\n" + prompt, return_tensors="pt").to(model.device)
            streamer = TextIteratorStreamer(tokenizer)
            thread = Thread(target=model.generate, kwargs={"inputs": input_ids, "max_new_tokens": 512, "streamer": streamer})
            thread.start()
            for token in streamer:
                yield f"<span style='font-family:Tw Cen MT'>{token}</span>"

    except Exception as e:
        yield f"<span style='font-family:Tw Cen MT'>⚠️ Error: {str(e)}</span>"

# Intro message per model
if not st.session_state.intro_done or st.session_state.current_model != model_alias:
    intro = {
        "EE Smartest Agent": "Hi, I'm **EE**, the proactive Insp-based engineer with strong capabilities in KPI analysis across B17. Ask me anything technical!",
        "JI Divine Agent": "Hello, I'm **JI**, trained with reasoning on Insp-based tasks and technical knowledge. I'm here to help!",
        "EdJa-Valonys": "Welcome! I'm **EdJa-Valonys**, optimized with GS's, GM's and CR's for rapid technical inference."
    }.get(model_alias, "")
    st.session_state.chat_history.append({"role": "assistant", "content": intro})
    st.session_state.intro_done = True
    st.session_state.current_model = model_alias

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me about the inspection reports or forecast progress..."):
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
