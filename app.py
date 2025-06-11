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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

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
        options=["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "HuggingFace RAG"],
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
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

# --- RAG Components ---
@st.cache_resource
def load_embedding_model():
    """Load Hugging Face embedding model for document processing"""
    return SentenceTransformer('all-MiniLM-L6-v2')  # Efficient and effective for embeddings

@st.cache_resource
def load_hf_model(selected_model):
    """Load Hugging Face model and tokenizer"""
    if selected_model == "Qwen":
        model_id = "amiguel/GM_Qwen1.8B_Finetune"
    elif selected_model == "Llama3":
        model_id = "amiguel/Llama3_8B_Instruct_FP16"
    else:
        model_id = "amiguel/GM_Mistral7B_Finetune"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    return model, tokenizer, model_id

def chunk_document(text, max_chunk_size=512):
    """Split document into smaller chunks for embedding"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_vector_index(chunks, embedding_model):
    """Create FAISS index from document chunks"""
    embeddings = embedding_model.encode(chunks, batch_size=32, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

def retrieve_relevant_chunks(query, embedding_model, index, chunks, top_k=3):
    """Retrieve top-k relevant document chunks for a query"""
    query_embedding = embedding_model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [chunks[idx] for idx in indices[0]]

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

# Process file upload and create embeddings
if uploaded_file and not st.session_state.file_context:
    st.session_state.file_context = parse_file(uploaded_file)
    if st.session_state.file_context:
        embedding_model = load_embedding_model()
        st.session_state.document_chunks = chunk_document(st.session_state.file_context)
        st.session_state.vector_index, st.session_state.document_chunks = create_vector_index(
            st.session_state.document_chunks, embedding_model
        )
        st.sidebar.success("✅ Document loaded and embeddings created")

def generate_hf_response(prompt, model, tokenizer, retrieved_chunks=None):
    """Generate response using Hugging Face model with RAG context"""
    context = "\n".join(retrieved_chunks) if retrieved_chunks else ""
    system_prompt = (
        f"You are DigiTwin, a digital expert and senior topside engineer. Use the following document context to provide accurate responses:\n{context}"
        if context else
        "You are DigiTwin, a digital expert and senior topside engineer. Provide accurate and concise responses."
    )
    
    inputs = tokenizer(
        f"{system_prompt}\nUser: {prompt}\nAssistant: ", return_tensors="pt", truncation=True, max_length=2048
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant: ")[-1]

def generate_response(prompt):
    """Generate AI response with RAG and bio fallback"""
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
            "content": f"Expert technical assistant. Current document context:\n{st.session_state.file_context}"
        } if st.session_state.file_context else {
            "role": "system",
            "content": "Expert technical assistant. Be concise and professional."
        }]
        
        messages.append({"role": "user", "content": prompt})
        
        start = time.time()
        
        if model_alias == "HuggingFace RAG":
            # Load Hugging Face model
            model, tokenizer, model_id = load_hf_model("Llama3")  # Default to Llama3 for RAG
            embedding_model = load_embedding_model()
            
            # Retrieve relevant document chunks
            retrieved_chunks = retrieve_relevant_chunks(
                prompt, embedding_model, st.session_state.vector_index, st.session_state.document_chunks
            ) if st.session_state.vector_index else None
            
            # Generate response
            full_response = generate_hf_response(prompt, model, tokenizer, retrieved_chunks)
            
            # Simulate streaming for UI consistency
            for word in full_response.split():
                yield word + " "
                time.sleep(0.05)
            yield "\n\n📄 Retrieved Context:\n" + "\n".join(retrieved_chunks) if retrieved_chunks else ""
            
        elif model_alias == "EE Smartest Agent":
            # Grok API (unchanged)
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
            # DeepSeek API (unchanged)
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
            # Cerebras API (unchanged)
            client = Cerebras(
                api_key=os.getenv("CEREBRAS_API_KEY"),
            )
            
            response = client.chat.completions.create(
                messages=messages,
                model="llama-4-scout-17b-16e-instruct"
            )
            
            if hasattr(response.choices[0], 'message'):
                full_response = response.choices[0].message.content
            else:
                full_response = str(response.choices[0])
            
            for word in full_response.split():
                yield word + " "
                time.sleep(0.05)
            yield ""
        
        # Performance metrics
        input_tokens = len(prompt.split())
        output_tokens = len(full_response.split())
        input_cost = (input_tokens / 1000000) * 5
        output_cost = (output_tokens / 1000000) * 15
        total_cost_usd = input_cost + output_cost
        exchange_rate = 1160
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
    elif model_alias == "HuggingFace RAG":
        intro_message = """
        Greetings, I am **HuggingFace RAG Agent**! 📚
        Powered by Hugging Face models with Retrieval-Augmented Generation:
        - **Context-Aware**: Leverages document embeddings for precise answers
        - **Efficient Retrieval**: Uses FAISS for fast document search
        - **Engineering-Focused**: Tailored for technical queries
        Upload a document and ask me anything!
        """
    
    st.session_state.chat_history.append({"role": "assistant", "content": intro_message})
    st.session_state.model_intro_done = True
    st.session_state.current_model = model_alias

# Chat interface
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about documents or technical matters..."):
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
