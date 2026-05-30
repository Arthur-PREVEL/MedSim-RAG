import streamlit as st
import torch
import json
import os
import re
import gc
import asyncio
import threading
import tempfile
import edge_tts
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="MedSim UI - Llama3 Test", page_icon="🩺", layout="wide")
st.title("🩺 MedSim: High-Fidelity Clinical Simulator")

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
    gc.collect()
    torch.cuda.empty_cache()

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
    path = '../data/knowledge_base_extract.json'
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
    patho_list = [item['title'] for item in kb] if kb else ["Default"]
    selected_patho = st.selectbox("Select a clinical case (Patient is hidden):", patho_list)
    
    if selected_patho != st.session_state.pathology:
        st.session_state.pathology = selected_patho
        st.session_state.messages = []
        st.session_state.interview_over = False
        st.session_state.input_key = 0
        clear_vram()
        st.rerun()

    if st.button("🔄 Restart Interview"):
        st.session_state.messages = []
        st.session_state.interview_over = False
        st.session_state.input_key = 0
        clear_vram()
        st.rerun()
        
    st.divider()
    
    st.info("""🧪 **Llama-3 Judge Evaluation Active**
    
* **Patient Mode:** BioMistral-7B-DARE (VRAM)
* **Judge Mode:** Meta-Llama-3-8B-Instruct (VRAM)
* **Hearing (STT):** Whisper Tiny (CPU)
* **Voice (TTS):** Edge-TTS (Network)""")

# ==========================================
# 4. CHAT INTERFACE (Patient Mode: BioMistral)
# ==========================================

# Define custom avatars for the UI
avatar_map = {"user": "🔵", "assistant": "🟢"}

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

        # Apply the blue avatar to the immediate user message
        with st.chat_message("user", avatar="🔵"):
            st.markdown(prompt)

        # Apply the green avatar to the immediate assistant message
        with st.chat_message("assistant", avatar="🟢"):
            with st.spinner("Patient is thinking and speaking..."):
                p_model, p_tokenizer = load_model_dynamic("BioMistral/BioMistral-7B-DARE")

                fiche = next((item for item in kb if item['title'] == st.session_state.pathology), None)
                rag_context = fiche['sections'].get('History and Physical', '') if fiche else ""
                
                chat_history = "\n".join([f"{'Doctor' if m['role']=='user' else 'Patient'}: {m['content']}" for m in st.session_state.messages])
                
                patient_prompt = f"""[INST] You are a 70-year-old patient. You are NOT a doctor.
                Describe your symptoms naturally based on this medical data: {rag_context}. 
                NEVER give the exact name of your disease.
                HISTORY:
                {chat_history}
                [/INST]"""

                inputs = p_tokenizer(patient_prompt, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    outputs = p_model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, pad_token_id=p_tokenizer.eos_token_id)
                
                response = p_tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
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
# 5. TRANSITION AND EVALUATION (Judge Mode: Llama-3)
# ==========================================
if len(st.session_state.messages) > 0 and not st.session_state.interview_over:
    if st.button("🏁 End interview and submit diagnosis"):
        st.session_state.interview_over = True
        st.sidebar.empty()
        st.rerun()

if st.session_state.interview_over:
    st.divider()
    st.subheader("🎓 Virtual Professor Evaluation")
    
    final_diagnosis = st.text_input("What is your final diagnosis?")
    
    if st.button("Evaluate my diagnosis") and final_diagnosis:
        with st.spinner("Purging Patient from VRAM & Initializing Llama-3 Professor..."):
            
            j_model, j_tokenizer = load_model_dynamic("meta-llama/Meta-Llama-3-8B-Instruct")
            
            fiche = next((item for item in kb if item['title'] == st.session_state.pathology), None)
            ground_truth = json.dumps(fiche['sections']) if fiche else ""
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

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ]
            
            prompt_text = j_tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )

            inputs = j_tokenizer(prompt_text, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = j_model.generate(
                    **inputs, 
                    max_new_tokens=500, 
                    temperature=0.2, 
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
                    st.write(fiche['sections'])
            else:
                st.error("JSON parsing failed. Llama-3 deviated from the formatting template. Raw text output:")
                st.code(raw_eval)