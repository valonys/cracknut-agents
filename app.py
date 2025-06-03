
import streamlit as st
import requests
import json
import os
import time
from dotenv import load_dotenv
import PyPDF2
from docx import Document
import pandas as pd
import openai  # For DeepSeek API
from cerebras.cloud.sdk import Cerebras  # For Cerebras API
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
load_dotenv()

# Configure Avatars
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# Preconfigured bio response
ATALIBA_BIO = """
**I am Ataliba Miguel's Digital Twin** 🤖

**Background:**
- 🎓 Mechanical Engineering (BSc)
- ⛽ Oil & Gas Engineering (MSc Specialization)
- 🔧 17+ years in Oil & Gas Industry
- 🔍 Current: Topside Inspection Methods Engineer @ TotalEnergies
- 🤖 AI Practitioner Specialist
- 🚀 Founder of ValonyLabs (AI solutions for industrial corrosion, retail analytics, and KPI monitoring)

**Capabilities:**
- Technical document analysis
- Engineering insights
- AI-powered problem solving
- Cross-domain knowledge integration

Ask me about engineering challenges, AI applications, or industry best practices!
"""

# Configure UI
st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/tw-cen-mt');
    * { font-family: 'Tw Cen MT', sans-serif; }
    .st-emotion-cache-1y4p8pa { padding: 2rem 1rem; }
    </style>
    """, unsafe_allow_html=True)
st.title("🚀 Ataliba o Agent Nerdx 🚀")

# Model selection in sidebar
with st.sidebar:
    st.header("⚙️ Model Selection")
    model_alias = st.selectbox(
        "Choose your AI Agent",
        options=["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "GM_Qwen1.8B_Finetune"],
        index=0,  # Default to Grok
        help="Select the AI model for your session."
    )

    st.header("📁 Document Hub")
    uploaded_file = st.file_uploader("Upload technical documents", type=["pdf", "docx", "xlsx", "xlsm"])

# Session state initialization
if "file_context" not in st.session_state:
    st.session_state.file_context = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_intro_done" not in st.session_state:
    st.session_state.model_intro_done = False
if "current_model" not in st.session_state:
    st.session_state.current_model = None

def parse_file(file):
    """Process uploaded file and return text content"""
    try:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() for page in reader.pages])
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            df = pd.read_excel(file)
            return df.to_string()
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Process file upload
if uploaded_file and not st.session_state.file_context:
    st.session_state.file_context = parse_file(uploaded_file)
    if st.session_state.file_context:
        st.sidebar.success("✅ Document loaded successfully")

def generate_response(prompt, model_alias):
    try:
        messages = [{
            "role": "system",
            "content": f"Expert technical assistant. Current document:\n{st.session_state.file_context}"
        } if st.session_state.file_context else {
            "role": "system",
            "content": "Expert technical assistant. Be concise and professional."



            
