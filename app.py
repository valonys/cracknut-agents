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
        options=["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys"],
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

def generate_response(prompt):
    """Generate AI response with bio fallback"""
    # Check for Ataliba-related questions
    bio_triggers = [
        'who are you', 'ataliba', 'yourself', 'skilled at', 
        'background', 'experience', 'valonylabs', 'totalenergies'
    ]
    
    if any(trigger in prompt.lower() for trigger in bio_triggers):
        for line in ATALIBA_BIO.split('\n'):
            yield line + '\n'
            time.sleep(0.1)
        return

    try:
        messages = [{
            "role": "system",
            "content": f"Expert technical assistant. Current document:\n{st.session_state.file_context}"
        } if st.session_state.file_context else {
            "role": "system",
            "content": "Expert technical assistant. Be concise and professional."
        }]
        
        messages.append({"role": "user", "content": prompt})
        
        start = time.time()
        
        if model_alias == "EE Smartest Agent":
            # Grok API
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-beta",
                    "messages": messages,
                    "temperature": 0.2,
                    "stream": True
                },
                stream=True
            )
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = line.decode('utf-8').replace('data: ', '')
                    if chunk == '[DONE]': break
                    try:
                        data = json.loads(chunk)
                        delta = data['choices'][0]['delta'].get('content', '')
                        full_response += delta
                        yield delta
                    except:
                        continue
            
        elif model_alias == "JI Divine Agent":
            # DeepSeek API
            client = openai.OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.sambanova.ai/v1",
            )
            
            response = client.chat.completions.create(
                model="DeepSeek-R1-Distill-Llama-70B",
                messages=messages,
                temperature=0.1,
                top_p=0.1,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    content = content.replace("<think>", "").replace("</think>", "")
                    full_response += content
                    yield content
        
        elif model_alias == "EdJa-Valonys":
            # Cerebras API (non-streaming implementation)
            client = Cerebras(
                api_key=os.getenv("CEREBRAS_API_KEY"),
            )
            
            response = client.chat.completions.create(
                messages=messages,
                model= " " #"amiguel/GM_Qwen1.8B_Finetune" #"llama-4-scout-17b-16e-instruct"
            )
            
            # Handle Cerebras response format
            if hasattr(response.choices[0], 'message'):
                full_response = response.choices[0].message.content
            else:
                full_response = str(response.choices[0])
            
            # Simulate streaming for UI consistency
            for word in full_response.split():
                yield word + " "
                time.sleep(0.05)
            yield ""  # Final yield
        
        # Performance metrics
        input_tokens = len(prompt.split())
        output_tokens = len(full_response.split())
        input_cost = (input_tokens / 1000000) * 5  # Grok pricing example
        output_cost = (output_tokens / 1000000) * 15  # Grok pricing example
        total_cost_usd = input_cost + output_cost
        exchange_rate = 1160  # USD to AOA
        total_cost_aoa = total_cost_usd * exchange_rate
        speed = output_tokens / (time.time() - start)
        yield f"\n\n🔑 Input Tokens: {input_tokens} | Output Tokens: {output_tokens} | 🕒 Speed: {speed:.1f}t/s | 💰 Cost (USD): ${total_cost_usd:.4f} | 💵 Cost (AOA): {total_cost_aoa:.4f}"
        
    except Exception as e:
        yield f"⚠️ API Error: {str(e)}"

# Model-specific introductions
if not st.session_state.model_intro_done or st.session_state.current_model != model_alias:
    if model_alias == "EE Smartest Agent":
        intro_message = """
        Hi, I am **EE**, the Double E Agent! 🚀

        My creator considers me a Double E agent because I am:
        - **Pragmatic**: I solve problems efficiently
        - **Innovative**: My reasoning capabilities go beyond human imagination
        - **Smart**: I am damn smarter than most systems out there

        How can I assist you today?
        """
    elif model_alias == "JI Divine Agent":
        intro_message = """
        Hi, I am **JI**, the Divine Agent! ✨

        My creator considers me a Divine Agent because I am:
        - **Gifted**: Trained to implement advanced reasoning
        - **Quasi-Human**: I mimic human intelligence and rational thinking
        - **Divine**: My capabilities are unparalleled

        How may I assist you today?
        """
    elif model_alias == "EdJa-Valonys":
        intro_message = """
        Greetings, I am **EdJa-Valonys**! ⚡

        The cutting-edge Cerebras-powered agent with:
        - **Lightning-fast inference**: Optimized for speed and efficiency
        - **Precision engineering**: Built on Llama-4 architecture
        - **Industrial-grade performance**: Designed for technical excellence

        What challenge can I help you solve today?
        """
    
    st.session_state.chat_history.append({"role": "assistant", "content": intro_message})
    st.session_state.model_intro_done = True
    st.session_state.current_model = model_alias

# Chat interface
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about documents or technical matters..."):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""
        
        for chunk in generate_response(prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
