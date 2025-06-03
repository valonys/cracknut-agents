import streamlit as st
import requests
import json
import os
import time
from dotenv import load_dotenv
import PyPDF2
from docx import Document
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

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
        index=0,
        help="Select the AI model for your session."
    )

    st.header("📁 Document Hub")
    uploaded_file = st.file_uploader("Upload technical documents", type=["pdf", "docx", "xlsx", "xlsm"])

    # Device selection
    device = st.selectbox(
        "Select device",
        options=["cuda" if torch.cuda.is_available() else "cpu", "cpu"],
        index=0,
        help="Select the device to run the model on (GPU recommended for larger models)"
    )

# Hugging Face model configurations
MODEL_CONFIGS = {
    "EE Smartest Agent": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "description": "Mistral 7B Instruct - A powerful 7B parameter model with strong reasoning capabilities"
    },
    "JI Divine Agent": {
        "repo_id": "meta-llama/Llama-2-70b-chat-hf",
        "description": "Llama 2 70B Chat - A large model with excellent conversational abilities"
    },
    "EdJa-Valonys": {
        "repo_id": "Qwen/Qwen1.5-1.8B-Chat",
        "description": "Qwen 1.8B Chat - A compact yet capable model optimized for chat"
    }
}

# Session state initialization
if "file_context" not in st.session_state:
    st.session_state.file_context = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_intro_done" not in st.session_state:
    st.session_state.model_intro_done = False
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "model_pipeline" not in st.session_state:
    st.session_state.model_pipeline = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

def load_model(model_name):
    """Load the selected Hugging Face model"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    repo_id = MODEL_CONFIGS[model_name]["repo_id"]
    
    # Check if we need to load a new model
    if st.session_state.current_model != model_name:
        st.sidebar.info(f"Loading {model_name} ({repo_id})...")
        
        # Clear previous model if exists
        if st.session_state.model_pipeline is not None:
            del st.session_state.model_pipeline
            del st.session_state.tokenizer
            torch.cuda.empty_cache()
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                device_map="auto",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            
            st.session_state.tokenizer = tokenizer
            st.session_state.model_pipeline = pipe
            st.session_state.current_model = model_name
            
            st.sidebar.success(f"Successfully loaded {model_name}")
            return pipe
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {str(e)}")
            return None
    else:
        return st.session_state.model_pipeline

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
        # Load the selected model
        pipe = load_model(model_alias)
        if pipe is None:
            yield "⚠️ Model failed to load. Please try again or select a different model."
            return
        
        # Prepare messages
        system_message = {
            "role": "system",
            "content": f"Expert technical assistant. Current document:\n{st.session_state.file_context}"
        } if st.session_state.file_context else {
            "role": "system",
            "content": "Expert technical assistant. Be concise and professional."
        }
        
        user_message = {"role": "user", "content": prompt}
        
        # Format messages according to model requirements
        if "mistral" in MODEL_CONFIGS[model_alias]["repo_id"].lower():
            messages = f"<s>[INST] {system_message['content']}\n{user_message['content']} [/INST]"
        elif "llama" in MODEL_CONFIGS[model_alias]["repo_id"].lower():
            messages = [
                {"role": "system", "content": system_message['content']},
                {"role": "user", "content": user_message['content']}
            ]
            messages = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
        else:  # Default format for other models
            messages = f"System: {system_message['content']}\nUser: {user_message['content']}\nAssistant:"
        
        start = time.time()
        
        # Generate response
        response = pipe(
            messages,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id,
            streamer=None  # Could implement proper streaming in future
        )
        
        # Process response based on model type
        if isinstance(response, list):
            full_response = response[0]['generated_text'][len(messages):]
        else:
            full_response = response['generated_text'][len(messages):]
        
        # Clean up response
        full_response = full_response.split("</s>")[0].strip()
        
        # Simulate streaming for better UX
        for word in full_response.split():
            yield word + " "
            time.sleep(0.05)
        
        # Performance metrics
        input_tokens = len(pipe.tokenizer.encode(messages))
        output_tokens = len(pipe.tokenizer.encode(full_response))
        speed = output_tokens / (time.time() - start)
        
        yield f"\n\n🔑 Input Tokens: {input_tokens} | Output Tokens: {output_tokens} | 🕒 Speed: {speed:.1f}t/s"
        
    except Exception as e:
        yield f"⚠️ Error generating response: {str(e)}"

# Model-specific introductions
if not st.session_state.model_intro_done or st.session_state.current_model != model_alias:
    if model_alias == "EE Smartest Agent":
        intro_message = """
        Hi, I am **EE**, the Double E Agent! 🚀
        Powered by Mistral-7B-Instruct from Hugging Face.

        My creator considers me a Double E agent because I am:
        - **Pragmatic**: I solve problems efficiently
        - **Innovative**: My reasoning capabilities go beyond human imagination
        - **Smart**: I am damn smarter than most systems out there

        How can I assist you today?
        """
    elif model_alias == "JI Divine Agent":
        intro_message = """
        Hi, I am **JI**, the Divine Agent! ✨
        Powered by Llama-2-70b-chat from Hugging Face.

        My creator considers me a Divine Agent because I am:
        - **Gifted**: Trained to implement advanced reasoning
        - **Quasi-Human**: I mimic human intelligence and rational thinking
        - **Divine**: My capabilities are unparalleled

        How may I assist you today?
        """
    elif model_alias == "EdJa-Valonys":
        intro_message = """
        Greetings, I am **EdJa-Valonys**! ⚡
        Powered by Qwen1.5-1.8B-Chat from Hugging Face.

        The cutting-edge open-source agent with:
        - **Lightning-fast inference**: Optimized for speed and efficiency
        - **Precision engineering**: Built on advanced architecture
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
