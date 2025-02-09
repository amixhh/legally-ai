import os
import json
import pickle
import numpy as np
import faiss
import streamlit as st
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
    
)
from peft import PeftModel
import torch
import torch.multiprocessing as mp
import time
import base64
import io
import docx
import PyPDF2
from pymongo import MongoClient
import jwt
import requests
from datetime import datetime
import bcrypt
import secrets
from dotenv import load_dotenv

mp.set_start_method("spawn", force=True)

# Configuration 
MODEL_NAME = "amixh/llama7b-legallyai-docsum"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_FILE = "legal_index.faiss"
DATA_FILE = "legal_data.pkl"
AUTH_SECRET = os.getenv('AUTH_SECRET')
CASES_DIR = "\streamlit-app\data"

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32
)

# Force CPU usage and disable MPS/CUDA
torch.backends.mps.is_available = lambda: False
torch.cuda.is_available = lambda: False
device = "cpu"

# Create offload directory if it doesn't exist
OFFLOAD_DIR = "offload_dir"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Legal Analysis AI",
    layout="wide",
    initial_sidebar_state="auto"
)

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv('MONGODB_URI')
NEXTAUTH_SECRET = os.getenv('NEXTAUTH_SECRET')
NEXTAUTH_URL = os.getenv('NEXT_AUTH_URL')

# Custom CSS
st.markdown("""
    <style>
    .chat-message {
        border: 1px solid #444444;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #2C2F33;
        color: black;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    .stTextArea textarea {
        min-height: 80px;
    }
    .upload-section {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px;
        border-top: 1px solid #ddd;
        margin-top: 20px;
    }
    .pin-button {
        background: none;
        border: none;
        font-size: 16px;
        cursor: pointer;
        padding: 5px;
        transition: transform 0.2s;
    }
    .pin-button:hover {
        transform: scale(1.1);
    }
    /* Style for the file input */
    .file-input {
        display: none;
    }
    /* Container for the upload elements */
    .upload-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
    }
    .main-title {
        text-align: center;
        font-weight: bold;
        color: black;
        margin-bottom: 2rem;
    }
    /* Center all text and elements */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        color: black !important;
    }
    /* Make all text black */
    .stMarkdown, .stText, .stTextInput, .stTextArea, button, .stButton {
        color: black !important;
    }
    div[data-testid="stForm"] {
        text-align: center;
    }
    .pin-upload-section {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .pin-emoji {
        font-size: 16px;
        cursor: pointer;
        margin-right: 5px;
    }
    .upload-label {
        font-size: 14px;
        color: black;
    }
    .stFileUploader {
        display: block !important;
        margin-bottom: 1rem;
    }
    /* Style for upload box */
    .uploadedFile {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    
    /* Make sure the upload text is visible */
    .st-emotion-cache-1gulkj5 {
        display: block !important;
        color: black !important;
    }
    
    /* Style for drag and drop area */
    .st-emotion-cache-1v04i6g {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        background-color: #fafafa;
    }
    </style>
""", unsafe_allow_html=True)

# Simplified chat history initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def login_form():
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if user_manager.verify_user(email, password):
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.success("Successfully logged in!")
                st.experimental_rerun()

                # After user logs in
                chat_history = user_manager.get_chat_history(st.session_state.user_email)
                for entry in chat_history:
                    st.session_state.chat_history.append(entry)
            else:
                st.error("Invalid credentials")

def signup_form():
    with st.form("signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign Up")
        
        if submit:
            if user_manager.create_user(email, password):
                st.success("Account created! Please login.")
            else:
                st.error("Email already exists")

@st.cache_resource
def load_model():
    try:
        with st.spinner('Loading model... This might take a few minutes...'):
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(device).eval()
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=128,
                device=device
            )
            
            return pipe
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Detailed error: {type(e).__name__} - {str(e)}")
        return None

def load_cases(directory: str) -> List[Dict]:
    cases = []
    for file in Path(directory).glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    case = json.load(f)
                    cases.append({
                        "id": file.stem,
                        "text": "\n".join(case.get("Input", [])),  # Join text chunks
                        "title": case.get("case_name", "Untitled Case")
                    })
                except json.JSONDecodeError as je:
                    st.error(f"JSON syntax error in file {file.name}: {str(je)}")
                    # Optionally, print the problematic line
                    f.seek(0)  # Go back to start of file
                    lines = f.readlines()
                    if je.lineno - 1 < len(lines):
                        st.code(f"Error near line {je.lineno}:\n{lines[je.lineno - 1]}")
        except Exception as e:
            st.error(f"Error reading file {file.name}: {str(e)}")
    
    if not cases:
        st.warning(f"No valid JSON files found in {directory}")
    
    return cases

def build_faiss_index(cases: List[Dict]):
    """Build index from case texts"""
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [case["text"] for case in cases]
    
    embeddings = encoder.encode(texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(DATA_FILE, 'wb') as f:
        pickle.dump([(c["title"], c["text"]) for c in cases], f)

def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve relevant case context"""
    start_time = time.time()
    
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(FAISS_INDEX_FILE)
    
    with open(DATA_FILE, 'rb') as f:
        cases = pickle.load(f)
    
    # Get more results than needed to filter duplicates
    query_embedding = encoder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, k * 2)  # Get more results
    
    # Filter unique cases
    seen_cases = set()
    unique_cases = []
    
    for idx in indices[0]:
        case_title = cases[idx][0]
        if case_title not in seen_cases and len(unique_cases) < k:
            seen_cases.add(case_title)
            unique_cases.append((case_title, cases[idx][1]))
    
    context = "\n\n".join(
        f"Case: {case[0]}\nText: {case[1][:1000]}..."
        for case in unique_cases
    )
    
    st.info(f"Facebook AI Similarity Search (FAISS) took {time.time() - start_time:.2f} seconds")
    return context

def generate_response(query: str, context: str, pipeline=None) -> str:
    """Generate legal analysis"""
    if pipeline is None:
        return "Error: Model not loaded properly."
        
    prompt = f"""Analyze this legal context and answer the question:
    
{context}

Question: {query}
Detailed Analysis:"""
    
    try:
        with torch.no_grad():
            response = pipeline(
                prompt,
                return_full_text=False,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1,
                max_new_tokens=512
            )[0]['generated_text']
            return response
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return f"Error generating response: {str(e)}"

def read_file_content(uploaded_file):
    """Read content from uploaded file (txt, pdf, or docx)"""
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'txt':
            # Read text file
            return uploaded_file.getvalue().decode('utf-8')
            
        elif file_type == 'pdf':
            # Read PDF file
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
            return text
            
        elif file_type == 'docx':
            # Read Word document
            doc = docx.Document(uploaded_file)
            text = ''
            for paragraph in doc.paragraphs:
                text += paragraph.text + '\n'
            return text
            
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def summarize_document(text: str, pipeline=None) -> str:
    """Document summarization"""
    if pipeline is None:
        return "Error: Model not loaded properly."
        
    prompt = f"""Please provide a comprehensive summary of this legal document:

{text[:3000]}  # Limit text to prevent token overflow

Summary:"""
    
    try:
        with torch.no_grad():
            response = pipeline(
                prompt,
                return_full_text=False,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1,
                max_new_tokens=512  # Longer output for summaries
            )[0]['generated_text']
            return response
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return f"Error generating summary: {str(e)}"

# Main interface
st.markdown("<h1 class='main-title'>Legally AI</h1>", unsafe_allow_html=True)

# Load the model
model_pipeline = load_model()

if model_pipeline is None:
    st.error("Failed to load the model. Please check system resources and try again.")
else:
    # Analysis and Summarization tabs
    tab1, tab2 = st.tabs(["Analysis", "Summarization"])
    
    # Analysis Tab
    with tab1:
        # File upload for analysis with explicit container
        st.write("Upload a document for context (optional)")
        uploaded_file = st.file_uploader(
            "",  # Empty label since we're using the write statement above
            type=['txt', 'pdf', 'docx'],
            key="qa_file_uploader",
            help="Drag and drop a file here or click to browse"
        )
        
        if uploaded_file:
            with st.spinner('Reading document...'):
                document_context = read_file_content(uploaded_file)
                if document_context:
                    st.session_state.current_document_context = document_context
                    st.success(f"Document loaded: {uploaded_file.name}")

        # Question input form
        with st.form(key="query_form", clear_on_submit=True):
            query = st.text_area("Enter your legal question:")
            submit_button = st.form_submit_button("Analyze Question")

        if submit_button and query:
            retrieved_context = retrieve_context(query)
            combined_context = retrieved_context
            
            if "current_document_context" in st.session_state:
                combined_context = f"Document Context:\n{st.session_state.current_document_context[:2000]}...\n\nRetrieved Cases:\n{retrieved_context}"
            
            analysis = generate_response(query, combined_context, model_pipeline)
            chat_entry = {
                "query": query,
                "response": analysis,
                "context": combined_context,
                "type": "qa",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.chat_history.append(chat_entry)

    # Summarization Tab
    with tab2:
        st.write("Upload a legal document for summarization")
        uploaded_file = st.file_uploader(
            "",  # Empty label since we're using the write statement above
            type=['txt', 'pdf', 'docx'],
            key="summary_file_uploader",
            help="Drag and drop a file here or click to browse"
        )
        
        if uploaded_file:
            with st.spinner('Reading document...'):
                document_text = read_file_content(uploaded_file)
                
                if document_text:
                    if st.button("Summarize Document"):
                        with st.spinner('Generating summary...'):
                            summary = summarize_document(document_text, model_pipeline)
                            
                            chat_entry = {
                                "query": f"Summarize document: {uploaded_file.name}",
                                "response": summary,
                                "context": document_text[:500] + "...",
                                "type": "summary",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state.chat_history.append(chat_entry)

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
            header = "Question" if entry["type"] == "qa" else "Document"
            response_header = "Answer" if entry["type"] == "qa" else "Summary"
            
            st.markdown(f"""
                <div style="border:1px solid #ddd; padding:10px; margin:10px 0; border-radius:5px;">
                    <div style="font-weight:bold;">{header} {i} - {entry["timestamp"]}</div>
                    <div style="margin:10px 0;">{entry["query"]}</div>
                    <div style="font-weight:bold;">{response_header}:</div>
                    <div style="margin:10px 0;">{entry["response"]}</div>
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"View Context for Entry {i}"):
                st.write(entry["context"])

        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")

    # Initialize system only if model loaded successfully
    if model_pipeline is not None:
        cases = load_cases(CASES_DIR)  # Load cases from the specified directory
        if cases:
            build_faiss_index(cases)  # Rebuild the index every time
        else:
            st.error(f"No cases loaded. Please check your JSON files in {CASES_DIR}")