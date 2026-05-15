import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import os

# ==========================================
# 1. ENGINE CONFIGURATION (V100 16GB)
# ==========================================
model_id = "BioMistral/BioMistral-7B-DARE"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print(f"🚀 Initializing BioMistral V3 (Simulation Mode)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa"
)

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_knowledge_base(file_path='knowledge_base_extract.json'):
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

KNOWLEDGE_BASE = load_knowledge_base()

# ==========================================
# 3. INTERACTIVE SIMULATION & EVALUATION
# ==========================================
def run_medical_sim(pathology_name):
    # Find the specific clinical record
    record = next((item for item in KNOWLEDGE_BASE if pathology_name.lower() in item['title'].lower()), None)
    if not record:
        print(f"❌ Pathology '{pathology_name}' not found in database.")
        return

    # Prepare data for different agents
    history_data = record['sections'].get('History and Physical', 'The patient feels unwell.')
    eval_data = record['sections'].get('Evaluation', 'Standard clinical protocols.')
    
    print(f"\n" + "="*60)
    print(f"🏥 CLINICAL SIMULATION START: {pathology_name.upper()}")
    print(f"Roleplay: You are the doctor. Talk to Jean (70yo patient).")
    print(f"Command: Type 'exit' to end the interview and receive your grade.")
    print("="*60 + "\n")

    transcript = ""
    
    # --- PHASE 1: THE PATIENT INTERVIEW (Jean) ---
    while True:
        user_input = input("👨‍⚕️ (Student): ")
        if user_input.lower() in ["exit", "quit", "stop"]: 
            break

        patient_prompt = f"""[INST] You are Jean, a 70-year-old patient. 
        INSTRUCTIONS: Use simple, everyday language. Do NOT use medical terms. 
        CONTEXT: You are suffering from these symptoms: {history_data}.
        RULE: Never mention the name of your disease. Describe how you feel.
        TRANSCRIPT: {transcript}
        QUESTION: {user_input} [/INST]"""

        inputs = tokenizer(patient_prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200, 
                temperature=0.7, 
                repetition_penalty=1.2, 
                no_repeat_ngram_size=3, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        print(f"\n👴 Jean: {response}\n")
        transcript += f"Doctor: {user_input}\nPatient: {response}\n"

    # --- PHASE 2: THE SUPERVISOR EVALUATION (Professor) ---
    print("\n" + "🎓" + "-"*58)
    print("  PROFESSOR EVALUATION & FEEDBACK")
    print("-"*60)
    
    eval_prompt = f"""[INST] You are a Medical Professor evaluating a medical student.
    CASE: {pathology_name}
    TRANSCRIPT OF THE INTERVIEW:
    {transcript}

    CLINICAL REFERENCE (GOLD STANDARD):
    {eval_data}

    YOUR GOAL: Critique the student's performance. Did they ask the right questions? 
    Did they identify the key symptoms? Provide a grade out of 20. [/INST]"""

    inputs = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    print("🧠 The Professor is analyzing your interview...")
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=450, 
            temperature=0.2,
            repetition_penalty=1.1
        )
    
    evaluation = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    print(f"\n{evaluation}\n")

if __name__ == "__main__":
    # Start the simulation with the pathology of your choice
    run_medical_sim("Osteoarthritis")
