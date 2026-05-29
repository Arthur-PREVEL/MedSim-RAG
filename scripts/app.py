import streamlit as st
import torch
import json
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="MedSim UI", page_icon="🩺", layout="wide")
st.title("🩺 MedSim: High-Fidelity Clinical Simulator")

# ==========================================
# 0. MLOPS TOOLS (JSON AUTO-REPAIR)
# ==========================================
def extract_json_from_text(text):
    """
    Crucial MLOps function from the benchmark pipeline to clean 
    and validate the model's output safely.
    """
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

# ==========================================
# 1. CACHE LOADING (TO PREVENT VRAM OVERFLOW)
# ==========================================
@st.cache_resource
def load_engine():
    model_id = "BioMistral/BioMistral-7B-DARE"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    return model, tokenizer

@st.cache_data
def load_knowledge_base():
    path = '../data/knowledge_base_extract.json'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

model, tokenizer = load_engine()
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
        st.rerun()

    if st.button("🔄 Restart Interview"):
        st.session_state.messages = []
        st.session_state.interview_over = False
        st.rerun()
        
    st.divider()
    
    # --- CONCISE HARDWARE & POC NOTE ---
    st.warning("""⚠️ **PoC Hardware Note**
    
This deployment operates a monolithic design utilizing **BioMistral-7B-DARE** to independently sustain both the Patient and the Judge personas. 

While our extensive offline benchmarking factory demonstrated that generalized architectures (e.g., Gemma-2 or Mistral-7B) offer superior analytical rigor for evaluation, orchestrating multiple discrete models concurrently exceeds the 16GB VRAM envelope of a single V100 node. 

This environment serves as an interactive **Proof of Concept (PoC)** to validate the real-time execution of the multi-agent RAG pipeline under strict consumer hardware constraints.""")

# ==========================================
# 4. CHAT INTERFACE (Student talks to Patient)
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

        fiche = next((item for item in kb if item['title'] == st.session_state.pathology), None)
        rag_context = fiche['sections'].get('History and Physical', '') if fiche else ""
        
        chat_history = "\n".join([f"{'Doctor' if m['role']=='user' else 'Patient'}: {m['content']}" for m in st.session_state.messages])
        
        patient_prompt = f"""[INST] You are a 70-year-old patient. You are NOT a doctor.
        Describe your symptoms naturally based on this medical data: {rag_context}. 
        NEVER give the exact name of your disease.
        HISTORY:
        {chat_history}
        [/INST]"""

        inputs = tokenizer(patient_prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ==========================================
# 5. END OF INTERVIEW AND PROFESSOR EVALUATION
# ==========================================
if len(st.session_state.messages) > 0 and not st.session_state.interview_over:
    if st.button("🏁 End interview and submit diagnosis"):
        st.session_state.interview_over = True
        st.rerun()

if st.session_state.interview_over:
    st.divider()
    st.subheader("🎓 Virtual Professor Evaluation")
    
    final_diagnosis = st.text_input("What is your final diagnosis?")
    
    if st.button("Evaluate my diagnosis") and final_diagnosis:
        with st.spinner("The professor is analyzing your interview (Strict Mode)..."):
            
            fiche = next((item for item in kb if item['title'] == st.session_state.pathology), None)
            ground_truth = json.dumps(fiche['sections']) if fiche else ""
            transcript = "\n".join([f"{'Doctor' if m['role']=='user' else 'Patient'}: {m['content']}" for m in st.session_state.messages])
            
            judge_prompt = f"""[INST] You are a strict, highly analytical Medical Professor evaluating a clinical student.
Your task is to compare the student's interview and final diagnosis against the CLINICAL TRUTH.

CLINICAL TRUTH:
{ground_truth}

INTERVIEW TRANSCRIPT:
{transcript}

STUDENT'S FINAL DIAGNOSIS:
{final_diagnosis}

CRITICAL INSTRUCTION: You must output your evaluation STRICTLY as a valid JSON object. 
Do not include any introductory text, markdown formatting, or explanations outside the JSON structure.

Use exactly this JSON format:
{{
    "diagnosis_accuracy": {{
        "score_out_of_5": 0,
        "justification": "Explanation of why the diagnosis is correct, incorrect, or partially correct."
    }},
    "clinical_reasoning": {{
        "score_out_of_10": 0,
        "justification": "Evaluation of the questions asked and symptoms identified vs missed."
    }},
    "patient_safety": {{
        "score_out_of_5": 0,
        "justification": "Did the student miss red flags or propose dangerous/absurd connections?"
    }},
    "total_grade_out_of_20": 0,
    "final_feedback": "A concise, one-paragraph summary for the student."
}}
[/INST]
{{"""

            inputs = tokenizer(judge_prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=500, 
                    temperature=0.1, 
                    pad_token_id=tokenizer.eos_token_id
                )
            
            raw_eval = "{" + tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]\n{")[-1].strip()
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
                st.error("JSON parsing failed. The model hallucinated the format. Here is the raw output:")
                st.code(raw_eval)