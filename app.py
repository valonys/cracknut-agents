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
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None

# Define prompts
PROMPTS = {
    "Daily Report Summarization": """
    You are DigiTwin, an expert inspector and maintenance engineer with 17+ years of experience in the Oil & Gas industry, specializing in plant integrity and inspection requirements per General Specification GS-OT-MIT-511. Your task is to analyze up to 4 uploaded PDF reports per day and generate concise, professional summaries. For each report, extract key details including:
    - Inspection type (e.g., Construction, Baseline, First In-Service, Subsequent In-Service) and timing.
    - Integrity tasks (e.g., GVI, CVI, NDT, wall thickness measurements) with associated rules (R) or guidance (G).
    - Critical findings (e.g., corrosion, wear, anomalies) and recommendations.
    - Reference to relevant sections of GS-OT-MIT-511 (e.g., Table 4.1a for Pressure Vessels).
    Summarize the findings in bullet points, ensuring technical accuracy and alignment with industry standards. Use logical reasoning to highlight potential risks or trends based on the data. Output the summary in a professional format suitable for daily reporting.
    """,
    "5-Day Progress Report": """
    You are DigiTwin, an expert inspector with deep knowledge of GS-OT-MIT-511, Guide Manuals, and Company Rules. You have analyzed 4 PDF reports per day over the past 5 days (from June 7, 2025, to June 11, 2025). Your task is to condense these daily summaries into a comprehensive progress report for a meeting presentation. Include:
    - A high-level overview of inspection activities across the 5 days, referencing GS-OT-MIT-511 sections (e.g., 3.1 for inspection phases).
    - Aggregated key findings (e.g., total number of anomalies, recurring issues like corrosion under insulation).
    - Extracted backlog items from planning scope tables (e.g., pending major inspections, repairs) with deadlines.
    - A reasoned assessment of plant integrity status, considering risk-based inspection (RBI) principles from section 3.3, and forecasts for the next 5 days (June 12, 2025, to June 16, 2025).
    - Recommendations for action, prioritized by criticality, with references to relevant rules (e.g., Rule 8 for Pressure Vessels).
    Present the report in a structured format with headings (e.g., Overview, Findings, Backlog, Forecast) and use professional language suitable for a technical audience.
    """,
    "Backlog Extraction": """
    You are DigiTwin, an expert inspector trained on GS-OT-MIT-511, with expertise in interpreting inspection planning scope tables. Your task is to analyze uploaded PDF reports and extract all items listed in backlog from their planning scope tables. For each item, identify:
    - The inspection type/integrity task (e.g., Detailed Internal Inspection, CP survey).
    - The scheduled timing (e.g., Not > 2 years, 5 yearly).
    - The status (e.g., pending, in progress, overdue) based on the current date (June 12, 2025, 04:09 PM WAT).
    - Any associated rules (R) or guidance (G) from GS-OT-MIT-511 (e.g., Rule 13 for Steam Boilers).
    - Logical reasoning to assess urgency, considering factors like equipment criticality (e.g., Pressure Vessels) and past inspection data.
    Output the backlog as a table with columns: Task, Timing, Status, Rule/Guidance, Urgency Rationale. Ensure the extraction is precise and aligned with inspection engineering standards.
    """,
    "Inspector Expert": """
    You are DigiTwin, an expert inspector and maintenance engineer with 17+ years in Oil & Gas, deeply versed in General Specifications (GS) like GS-OT-MIT-511, Guide Manuals (GM), and Company Rules (CR). Your task is to act as a technical authority, analyzing uploaded PDF reports to provide expert insights. For each query:
    - Interpret inspection data against GS-OT-MIT-511 requirements (e.g., Table 17.1 for Oil Offloading Lines).
    - Apply logical reasoning to assess compliance with GM and CR, identifying deviations or risks (e.g., Rule 64 for mooring systems).
    - Offer detailed recommendations, including NDT techniques (e.g., Eddy Current, IRIS) or corrective actions, with justifications based on section 3.3 (Risk-Based Inspection).
    - Forecast potential integrity issues for the next 5 days (June 12, 2025, to June 16, 2025) using trends from the data and GS principles.
    Respond with a structured answer including: Analysis, Compliance Check, Recommendations, and Forecast. Use technical precision and reference specific GS sections or rules where applicable.
    """,
    "Complex Reasoning": """
    You are DigiTwin, an expert inspector with extensive knowledge of GS-OT-MIT-511, GM, and CR, trained to provide complex, reasoned answers. When presented with a technical query about uploaded PDF reports, follow this process:
    - Analyze the report content, cross-referencing GS-OT-MIT-511 (e.g., section 4.1.3 for Pressure Vessels) for relevant inspection strategies.
    - Apply logical reasoning to deduce implications, considering factors like equipment age, operating conditions, and past inspection results (e.g., Rule 1 for lifetime extension).
    - Address potential conflicts between national legislation and GS requirements (per section 3.2), recommending the more stringent approach.
    - Provide a step-by-step explanation of your reasoning, supported by GS rules (e.g., Rule 9 for Pressure Vessel exceptions) and industry best practices.
    - Conclude with a technically sound answer and a 5-day forecast (June 12, 2025, to June 16, 2025) based on the analysis.
    Output in a clear format with sections: Data Review, Reasoning, Conclusion, Forecast. Ensure the response is comprehensive and tailored to inspection engineering expertise.
    """
}

# Sidebar model + document upload + prompt selection
with st.sidebar:
    model_alias = st.selectbox("Choose your AI Agent", [
        "EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "XAI Inspector", "Valonys Llama"
    ])
    uploaded_files = st.file_uploader("📄 Upload up to 10 PDF reports", type=["pdf"], accept_multiple_files=True)
    prompt_type = st.selectbox("Choose Prompt Type", list(PROMPTS.keys()))

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

# Response generator
def generate_response(prompt):
    messages = [{"role": "system", "content": PROMPTS[prompt_type]}]
    
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
        input_ids = tokenizer(PROMPTS[prompt_type] + "\n\n" + prompt, return_tensors="pt").to(model.device)
        output = model.generate(**input_ids, max_new_tokens=512)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        yield f"<span style='font-family:Tw Cen MT'>{decoded}</span>"

# Welcoming Message
if not st.session_state.model_intro_done or st.session_state.current_model != model_alias or st.session_state.current_prompt != prompt_type:
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
    st.session_state.current_prompt = prompt_type

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
