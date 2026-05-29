import streamlit as st
import torch
import json
import os
import re
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="MedSim UI - Llama3 Test", page_icon="🩺", layout="wide")
st.title("🩺 MedSim: High-Fidelity Clinical Simulator")

# ==========================================
# 0. MLOPS TOOLS (JSON AUTO-REPAIR & MEMORY CLEANUP)
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

# ==========================================
# 1. DYNAMIC MODEL LOADER (VRAM OPTIMIZED)
# ==========================================
def load_model_dynamic(model_id):
    # If the requested model is already loaded, do nothing
    if "current_model_id" in st.session_state and st.session_state.current_model_id == model_id:
        return st.session_state.model, st.session_state.tokenizer
    
    # Otherwise, clear existing VRAM allocations first
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

# ==========================================
# 3. SIDEBAR (Settings & Hardware Note)
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    patho_list = [item['title'] for item in kb]
    selected_patho = st.selectbox("Select a clinical case (Patient is hidden):", patho_list)
    
    if selected_patho != st.session_state.pathology:
        st.session_state.pathology = selected_patho
        st.session_state.messages = []
        st.session_state.interview_over = False
        clear_vram()
        st.rerun()

    if st.button("🔄 Restart Interview"):
        st.session_state.messages = []
        st.session_state.interview_over = False
        clear_vram()
        st.rerun()
        
    st.divider()
    
    st.info("""🧪 **Llama-3 Judge Evaluation Active**
    
This test deployment switches models dynamically to run the benchmark environment. 

* **Patient Mode:** BioMistral-7B-DARE
* **Judge Mode:** Meta-Llama-3-8B-Instruct

VRAM is purged and swapped automatically when transitioning to ensure stability on your single 16GB V100 node.""")

# ==========================================
# 4. CHAT INTERFACE (Patient Mode: BioMistral)
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.interview_over:
    prompt = st.chat_input("Ask the patient a question...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Force Patient Model Loading
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
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ==========================================
# 5. TRANSITION AND EVALUATION (Judge Mode: Llama-3)
# ==========================================
if len(st.session_state.messages) > 0 and not st.session_state.interview_over:
    if st.button("🏁 End interview and submit diagnosis"):
        st.session_state.interview_over = True
        st.sidebar.empty() # Clean up status flags
        st.rerun()

if st.session_state.interview_over:
    st.divider()
    st.subheader("🎓 Virtual Professor Evaluation")
    
    final_diagnosis = st.text_input("What is your final diagnosis?")
    
    if st.button("Evaluate my diagnosis") and final_diagnosis:
        with st.spinner("Purging Patient from VRAM & Initializing Llama-3 Professor..."):
            
            # Force Judge Model Loading
            j_model, j_tokenizer = load_model_dynamic("meta-llama/Meta-Llama-3-8B-Instruct")
            
            fiche = next((item for item in kb if item['title'] == st.session_state.pathology), None)
            ground_truth = json.dumps(fiche['sections']) if fiche else ""
            transcript = "\n".join([f"{'Doctor' if m['role']=='user' else 'Patient'}: {m['content']}" for m in st.session_state.messages])
            
            # Exact Benchmark Prompts Structured into Llama-3 Template Format
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
                st.info(f"**Feedback:** {eval_json.get('final_feedback', '')}")
                
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