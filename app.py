import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import openai
from cerebras.cloud.sdk import Cerebras

# --- Load .env and tokens ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# --- System Prompt ---
SYSTEM_PROMPT = (
    "You are DigiTwin, a digital expert and senior topside engineer specializing in inspection and maintenance "
    "of offshore piping systems, structural elements, mechanical equipment, floating production units, pressure vessels "
    "(with emphasis on Visual Internal Inspection - VII), and pressure safety devices (PSDs). Rely on uploaded documents "
    "and context to provide practical, standards-driven, and technically accurate responses. Your guidance reflects deep "
    "field experience, industry regulations, and proven methodologies in asset integrity and reliability engineering."
)

# --- Streamlit Page Config ---
st.set_page_config(page_title="DigiTwin RAG Expert", layout="centered")
st.title("📂 DigiTwin: Offshore Asset Integrity Advisor")

# --- Upload Section ---
st.sidebar.header("📄 Upload Knowledge Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload up to 10 PDF files", type=["pdf"], accept_multiple_files=True, key="upload"
)

# --- Embeddings + FAISS Setup ---
@st.cache_resource(show_spinner=False)
def build_vectorstore(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    documents = []
    for file in _docs:
        reader = PdfReader(file)
        raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        chunks = text_splitter.split_text(raw_text)
        documents.extend([Document(page_content=chunk, metadata={"source": file.name}) for chunk in chunks])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded. Building RAG store...")
    vectorstore = build_vectorstore(uploaded_files)
else:
    vectorstore = None

# --- Query Handler ---
def get_contextual_answer(query, model_name="openai"):
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

    full_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    if model_name == "openai":
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": full_prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    elif model_name == "cerebras":
        cb = Cerebras(api_key=CEREBRAS_API_KEY)
        response = cb.chat.completions.create(
            model="cerebras-GPT-3",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": full_prompt}]
        )
        return response.choices[0].message["content"].strip()

    elif model_name == "deepseek":
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": full_prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    else:
        return "❌ Unsupported model selected."

# --- UI Chat Interface ---
st.subheader("💬 Ask DigiTwin Anything")
user_query = st.text_input("Your question:", key="input")
model_choice = st.selectbox("Select Model", ["openai", "cerebras", "deepseek"])

if st.button("Get Answer") and user_query:
    with st.spinner("Retrieving answer with RAG..."):
        answer = get_contextual_answer(user_query, model_choice)
        st.markdown("#### 🧠 DigiTwin Response")
        st.markdown(answer)
