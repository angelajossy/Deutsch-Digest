import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deutsch-Digest",
    page_icon="🇩🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR UI ---
st.markdown("""
<style>
    /* Main Background gradient */
    .stApp {
        background: linear-gradient(to bottom right, #ffffff, #f0f2f6);
    }
    
    /* Header Styling */
    h1 {
        color: #1E1E1E;
        font-family: 'Helvetica', sans-serif;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Custom Button Styling */
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 28px; /* Increased padding for a bigger look */
        font-weight: bold;
        font-size: 16px;
        white-space: nowrap; /* <--- THIS FIXES THE TEXT WRAPPING */
        min-width: 200px;    /* <--- THIS MAKES THE BUTTON LONG & SLEEK */
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #cc0000;
        transform: scale(1.02);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        font-size: 16px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
MODEL_REPO = "Einmalumdiewelt/T5-Base_GNAD"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO)
    return tokenizer, model

try:
    tokenizer, model = load_model()
    translator = Translator()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# --- SESSION STATE ---
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

def clear_text():
    st.session_state.input_text = ""

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/b/ba/Flag_of_Germany.svg/1200px-Flag_of_Germany.svg.png", width=60)
    st.title("Deutsch-Digest")

# --- MAIN INTERFACE ---
st.title("German Summarizer")

# Input Area
text_input = st.text_area(
    "Input German Text", 
    height=250, 
    key="input_text",
    placeholder="Paste here"
)

# Button Columns
# UPDATED RATIO: [2, 5] gives the first button more space so it won't squeeze
col_btn1, col_btn2 = st.columns([2, 5])

with col_btn1:
    summarize_btn = st.button("Summarize & Verify") 
with col_btn2:
    clear_btn = st.button("Clear", on_click=clear_text)

if summarize_btn:
    if text_input and model_loaded:
        with st.spinner('Generating summary...'):
            try:
                # A. GENERATE
                input_ids = tokenizer(
                    "summarize: " + text_input,
                    return_tensors="pt",
                    max_length=1024, 
                    truncation=True
                )["input_ids"]

                output_ids = model.generate(
                    input_ids,
                    max_length=250,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
                
                summary_de = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # B. TRANSLATE
                translation = translator.translate(summary_de, src='de', dest='en')
                summary_en = translation.text
                
                # C. DISPLAY RESULTS
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### German Summary")
                    st.success(summary_de)
                    
                with col2:
                    st.markdown("### English Verification")
                    st.info(summary_en)
                    
            except Exception as e:
                st.error(f"Processing Error: {e}")
            
    elif not text_input:
        st.warning("Please enter some text first.")