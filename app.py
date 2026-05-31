import os
# Must be set before importing torch to prevent VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import streamlit as st
import torch
import json
import re
import gc
import asyncio
import threading
import tempfile
import edge_tts
from PIL import Image
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# FILE PATHS & MODEL REGISTRY SETUP
# ==========================================
# Assuming app.py is at the root and images are in /images/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")

ICON_PATH = os.path.join(IMG_DIR, "medsim_logo.ico")
LOGO_PATH = os.path.join(IMG_DIR, "medsim_logo.png")
PATIENT_ICON = os.path.join(IMG_DIR, "patient.png")
DOCTOR_ICON = os.path.join(IMG_DIR, "doctor.png")

# Centralized Model Registry
MODEL_REGISTRY = {
    "BioMistral (7B)": "BioMistral/BioMistral-7B-DARE",
    "Meta Llama-3 (8B)": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Mistral v0.3 (7B)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Google Gemma-2 (9B)": "google/gemma-2-9b-it",
    "Microsoft Phi-3 (3.8B)": "microsoft/Phi-3-mini-4k-instruct"
}

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
# Load the .ico file safely using PIL (fallback to emoji if missing during dev)
page_icon = Image.open(ICON_PATH) if os.path.exists(ICON_PATH) else "🩺"

st.set_page_config(page_title="MedSim UI", page_icon=page_icon, layout="wide")

# Display Logo and Title side-by-side
col_logo, col_title = st.columns([1, 11])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
with col_title:
    st.title("MedSim: High-Fidelity Clinical Simulator")

# ==========================================
# 0. MLOPS & AUDIO TOOLS 
# ==========================================
def extract_json_from_text(text):
    text = text.strip()
    if text.startswith('{') and not text.endswith('}'):
        text += '\n}'
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    return None

def clear_vram():
    if "model" in st.session_state:
        del st.session_state.model
    if "tokenizer" in st.session_state:
        del st.session_state.tokenizer
    
    # Force Python to destroy unreferenced objects
    gc.collect()
    
    # Empty PyTorch's cache and release it back to the OS
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# Threading hack to safely run async edge-tts inside Streamlit's sync environment
def generate_tts(text, voice):
    audio_data = b""
    def _run_async():
        async def _amain():
            communicate = edge_tts.Communicate(text, voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    nonlocal audio_data
                    audio_data += chunk["data"]
        asyncio.run(_amain())
        
    thread = threading.Thread(target=_run_async)
    thread.start()
    thread.join()
    return audio_data

# ==========================================
# 1. DYNAMIC MODEL LOADERS (VRAM & CPU)
# ==========================================
@st.cache_resource
def load_whisper_cpu():
    return WhisperModel("tiny.en", device="cpu", compute_type="int8")

whisper_model = load_whisper_cpu()

def load_model_dynamic(model_id):
    if "current_model_id" in st.session_state and st.session_state.current_model_id == model_id:
        return st.session_state.model, st.session_state.tokenizer
    
    st.write(f"⏳ Dynamic VRAM Swap: Loading {model_id}...")
    clear_vram()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.session_state.current_model_id = model_id
    return model, tokenizer

@st.cache_data
def load_knowledge_base():
    path = os.path.join(BASE_DIR, 'data/knowledge_base_extract.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

kb = load_knowledge_base()

# ==========================================
# 2. STATE MANAGEMENT (CHAT MEMORY)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pathology" not in st.session_state:
    st.session_state.pathology = kb[0]['title'] if kb else "Unknown"
if "interview_over" not in st.session_state:
    st.session_state.interview_over = False
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# ==========================================
# 3. SIDEBAR (Settings & Hardware Note)
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Clinical Case Selection
    patho_list = [item['title'] for item in kb] if kb else ["Default"]
    selected_patho = st.selectbox("Select a clinical case (Patient is hidden):", patho_list)
    
    st.divider()
    
    # Dynamic LLM Selection
    st.subheader("🧠 Multi-Agent Engine")
    
    # Defaults based on previous hardcoded values
    default_patient_idx = list(MODEL_REGISTRY.keys()).index("Meta Llama-3 (8B)")
    default_judge_idx = list(MODEL_REGISTRY.keys()).index("Google Gemma-2 (9B)")
    
    selected_patient_name = st.selectbox("🤖 Patient Model:", list(MODEL_REGISTRY.keys()), index=default_patient_idx)
    selected_judge_name = st.selectbox("⚖️ Judge Model:", list(MODEL_REGISTRY.keys()), index=default_judge_idx)
    
    patient_model_hf = MODEL_REGISTRY[selected_patient_name]
    judge_model_hf = MODEL_REGISTRY[selected_judge_name]

    # Reset behavior if pathology changes
    if selected_patho != st.session_state.pathology:
        st.session_state.pathology = selected_patho
        st.session_state.messages = []
        st.session_state.interview_over = False
        st.session_state.input_key = 0
        clear_vram()
        st.rerun()

    st.divider()

    if st.button("🔄 Restart Interview"):
        st.session_state.messages = []
        st.session_state.interview_over = False
        st.session_state.input_key = 0
        clear_vram()
        st.rerun()
        
    st.info(f"""🧪 **Current Hardware Allocation**
    
* **Patient Mode:** {selected_patient_name} (VRAM)
* **Judge Mode:** {selected_judge_name} (VRAM)
* **Hearing (STT):** Whisper Tiny (CPU)
* **Voice (TTS):** Edge-TTS (Network)""")

# ==========================================
# 4. CHAT INTERFACE (Patient Mode)
# ==========================================

# Define custom avatars
avatar_map = {
    "user": DOCTOR_ICON if os.path.exists(DOCTOR_ICON) else "🔵", 
    "assistant": PATIENT_ICON if os.path.exists(PATIENT_ICON) else "🟢"
}

# Show an onboarding prompt if the interview just started
if len(st.session_state.messages) == 0 and not st.session_state.interview_over:
    st.info("🚪 **The patient has just entered the room; start the conversation with a simple phrase such as:** *“Hello, sir, what brings you here today?”*")

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar_map.get(message["role"])):
        st.markdown(message["content"])
        if message.get("audio"):
            st.audio(message["audio"], format="audio/mp3", autoplay=message.get("autoplay", False))
            if message.get("autoplay"):
                message["autoplay"] = False

if not st.session_state.interview_over:
    
    audio_value = st.audio_input("🎤 Speak to the patient", key=f"audio_input_{st.session_state.input_key}")
    text_prompt = st.chat_input("⌨️ Or type your question...")
    
    prompt = text_prompt 
    
    if audio_value is not None:
        with st.spinner("🧠 Transcribing on CPU (Whisper)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_value.getvalue())
                tmp_path = tmp_file.name
            
            try:
                segments, _ = whisper_model.transcribe(tmp_path, beam_size=5)
                prompt = "".join([segment.text for segment in segments]).strip()
            finally:
                os.remove(tmp_path)
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Apply the doctor avatar
        with st.chat_message("user", avatar=avatar_map["user"]):
            st.markdown(prompt)

        # Apply the patient avatar
        with st.chat_message("assistant", avatar=avatar_map["assistant"]):
            with st.spinner(f"{selected_patient_name} is thinking and speaking..."):
                p_model, p_tokenizer = load_model_dynamic(patient_model_hf)

                fiche = next((item for item in kb if item['title'] == st.session_state.pathology), None)
                rag_context = fiche['sections'].get('History and Physical', '') if fiche else ""
                
                chat_history = "\n".join([f"{'Doctor' if m['role']=='user' else 'Patient'}: {m['content']}" for m in st.session_state.messages])
                
                # Combine System Instruction + Context into a single User message for template compatibility across models
                patient_instruction = f"""You are a 70-year-old patient. You are NOT a doctor.
Describe your symptoms naturally based on this medical data: {rag_context}. 
NEVER give the exact name of your disease.

HISTORY:
{chat_history}"""

                p_messages = [{"role": "user", "content": patient_instruction}]
                
                p_prompt_text = p_tokenizer.apply_chat_template(
                    p_messages, 
                    add_generation_prompt=True, 
                    tokenize=False
                )

                inputs = p_tokenizer(p_prompt_text, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    outputs = p_model.generate(
                        **inputs, 
                        max_new_tokens=250, # Increased to allow the patient to finish their thought
                        temperature=0.7, 
                        do_sample=True, 
                        repetition_penalty=1.1, # Added to prevent repetitive symptom looping
                        pad_token_id=p_tokenizer.eos_token_id
                    )
                
                # Accurately slice only the newly generated tokens
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                response = p_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                # 🛡️ SECURITY NET: Cleanly truncate the last sentence if cut off by the token limit
                if not response.endswith(('.', '!', '?', '"')):
                    # Find the last strong punctuation
                    match = re.search(r'(.*[.!?])', response, re.DOTALL)
                    if match:
                        response = match.group(1).strip()
                    else:
                        response += "..." # Fallback if no punctuation is found

                audio_bytes = generate_tts(response, voice="en-US-GuyNeural")
                
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "audio": audio_bytes,
            "autoplay": True
        })
        
        st.session_state.input_key += 1
        st.rerun()

# ==========================================
# 5. TRANSITION AND EVALUATION (Judge Mode)
# ==========================================
if len(st.session_state.messages) > 0 and not st.session_state.interview_over:
    if st.button("🏁 End interview and submit diagnosis"):
        st.session_state.interview_over = True
        st.sidebar.empty()
        st.rerun()

if st.session_state.interview_over:
    st.divider()
    st.subheader(f"🎓 Virtual Professor Evaluation ({selected_judge_name})")
    
    final_diagnosis = st.text_input("What is your final diagnosis?")
    
    if st.button("Evaluate my diagnosis") and final_diagnosis:
        with st.spinner(f"Purging Patient from VRAM & Initializing {selected_judge_name}..."):
            
            j_model, j_tokenizer = load_model_dynamic(judge_model_hf)
            
            fiche = next((item for item in kb if item['title'] == st.session_state.pathology), None)
            
            # VRAM OPTIMIZATION: Send only the essential symptoms, not the entire JSON record.
            ground_truth = fiche['sections'].get('History and Physical', '') if fiche else ""
            
            transcript = "\n".join([f"{'Doctor' if m['role']=='user' else 'Patient'}: {m['content']}" for m in st.session_state.messages])
            
            system_instruction = """You are a strict, highly analytical Medical Professor evaluating a clinical student.
Your task is to compare the student's interview and final diagnosis against the CLINICAL TRUTH.

CRITICAL INSTRUCTION: You must output your evaluation STRICTLY as a valid JSON object. 
Do not include any introductory text, markdown formatting, or explanations outside the JSON structure.

Use exactly this JSON format:
{
    "diagnosis_accuracy": {
        "score_out_of_5": 0,
        "justification": "Explanation of why the diagnosis is correct, incorrect, or partially correct."
    },
    "clinical_reasoning": {
        "score_out_of_10": 0,
        "justification": "Evaluation of the questions asked and symptoms identified vs missed."
    },
    "patient_safety": {
        "score_out_of_5": 0,
        "justification": "Did the student miss red flags or propose dangerous/absurd connections?"
    },
    "total_grade_out_of_20": 0,
    "final_feedback": "A concise, one-paragraph summary for the student."
}"""

            user_content = f"""### CLINICAL TRUTH:
{ground_truth}

### INTERVIEW TRANSCRIPT:
{transcript}

### STUDENT'S FINAL DIAGNOSIS:
{final_diagnosis}

Please provide your evaluation now."""

            # Combined content to ensure template compatibility across all models (like BioMistral)
            combined_content = f"{system_instruction}\n\n{user_content}"
            messages = [{"role": "user", "content": combined_content}]
            
            prompt_text = j_tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )

            inputs = j_tokenizer(prompt_text, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = j_model.generate(
                    **inputs, 
                    max_new_tokens=600, 
                    temperature=0.1, 
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=j_tokenizer.eos_token_id
                )
            
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            raw_eval = j_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            eval_json = extract_json_from_text(raw_eval)
            
            if eval_json:
                st.success(f"### Final Grade: {eval_json.get('total_grade_out_of_20', '?')}/20")
                
                feedback_text = eval_json.get('final_feedback', '')
                st.info(f"**Feedback:** {feedback_text}")
                
                audio_bytes = generate_tts(feedback_text, voice="en-US-AriaNeural")
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Diagnostic Accuracy (/5)", eval_json.get('diagnosis_accuracy', {}).get('score_out_of_5', '?'))
                    st.caption(eval_json.get('diagnosis_accuracy', {}).get('justification', ''))
                with col2:
                    st.metric("Clinical Reasoning (/10)", eval_json.get('clinical_reasoning', {}).get('score_out_of_10', '?'))
                    st.caption(eval_json.get('clinical_reasoning', {}).get('justification', ''))
                with col3:
                    st.metric("Patient Safety (/5)", eval_json.get('patient_safety', {}).get('score_out_of_5', '?'))
                    st.caption(eval_json.get('patient_safety', {}).get('justification', ''))
                    
                with st.expander("🔍 View Clinical Truth (RAG Reference)"):
                    if fiche:
                        st.write(fiche['sections'])
                    else:
                        st.write("No RAG reference available.")
            else:
                st.error("JSON parsing failed. The selected model deviated from the formatting template. Raw text output:")
                st.code(raw_eval)