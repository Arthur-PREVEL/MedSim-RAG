import os
import sys

# Add the parent directory to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Must be set before importing torch to prevent memory fragmentation OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.orchestrator import MedSimOrchestrator
import json

# ==========================================
# RESEARCH VARIABLES
# ==========================================
PATIENT_MODEL_ID = "BioMistral/BioMistral-7B-DARE"
DOCTOR_MODEL_ID = "BioMistral/BioMistral-7B-DARE" 
PATHOLOGY = "Osteoarthritis"
TURNS = 4

def load_agent(model_id):
    print(f"⌛ Loading model instance: {model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    return model, tokenizer

def run_simulation():
    # Pass the explicit path to the data file relative to the script
    orchestrator = MedSimOrchestrator(file_path='../data/knowledge_base_extract.json')
    
    # --- VRAM OPTIMIZATION ---
    # If the patient and doctor are the same model, only load it once!
    if PATIENT_MODEL_ID == DOCTOR_MODEL_ID:
        print("💡 VRAM Optimization: Using the same model for Patient & Doctor. Loading once.")
        shared_model, shared_tok = load_agent(PATIENT_MODEL_ID)
        patient_model, patient_tok = shared_model, shared_tok
        doctor_model, doctor_tok = shared_model, shared_tok
    else:
        # Initialize Dual Instances (Only if models are different)
        patient_model, patient_tok = load_agent(PATIENT_MODEL_ID)
        doctor_model, doctor_tok = load_agent(DOCTOR_MODEL_ID)

    patient_context = orchestrator.get_context(PATHOLOGY, "patient")
    transcript = []
    
    print(f"\n🚀 DUAL-AGENT SIMULATION: {PATHOLOGY.upper()}")
    print("-" * 50)
    
    current_message = "Hello, I am the physician attending to you today. What symptoms are you experiencing?"
    transcript.append({"role": "Doctor", "content": current_message})
    print(f"🩺 [Doctor]: {current_message}")

    # --- PHASE 1: CLINICAL INTERVIEW ---
    for i in range(TURNS):
        
        # --- PATIENT TURN ---
        patient_prompt = f"""[INST] You are Jean, a 70-year-old patient. 
        SYMPTOMS CONTEXT: {patient_context}
        RULES: 
        1. Use simple language (no medical terms like 'nodes' or 'locomotor').
        2. Describe your pain naturally.
        3. Do NOT repeat what the doctor says.
        4. ABSOLUTE RULE: You are the PATIENT, do NOT act as a doctor.
        QUESTION FROM DOCTOR: {current_message} [/INST]"""
        
        inputs = patient_tok(patient_prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            ids = patient_model.generate(**inputs, max_new_tokens=150, temperature=0.8, do_sample=True, pad_token_id=patient_tok.eos_token_id)
        
        current_message = patient_tok.decode(ids[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        # Aggressive scrubbing of AI formatting artifacts
        current_message = current_message.replace("Patient:", "").replace("Jean:", "").replace("Answer:", "").replace('"', '').strip()
        
        print(f"👴 [Patient]: {current_message}")
        transcript.append({"role": "Patient", "content": current_message})

        # --- PREVENT DANGLING DOCTOR QUESTION ---
        # If this is the last turn, stop here so the Patient gets the final word!
        if i == TURNS - 1:
            break

        # --- DOCTOR TURN ---
        doc_prompt = f"""[INST] You are a Doctor conducting a diagnostic interview.
        CURRENT INFO FROM PATIENT: {current_message}
        GOAL: Ask ONE brief question to help differentiate the patient's condition. Focus on ruling out other forms of arthritis.
        CRITICAL RULE: Output ONLY the question you want to ask. Do not write your internal thoughts. [/INST]"""
        
        inputs = doctor_tok(doc_prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            ids = doctor_model.generate(**inputs, max_new_tokens=120, temperature=0.3, do_sample=True, pad_token_id=doctor_tok.eos_token_id)
            
        current_message = doctor_tok.decode(ids[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        current_message = current_message.replace("Doctor:", "").strip()
        
        print(f"🩺 [Doctor]: {current_message}")
        transcript.append({"role": "Doctor", "content": current_message})

    # --- PHASE 2: FINAL DIAGNOSIS SYNTHESIS ---
    print("\n🧐 [Doctor is formulating the final diagnosis...]")
    
    full_history = "\n".join([f"{m['role']}: {m['content']}" for m in transcript])
    final_prompt = f"""[INST] Review the following medical interview history. 
    State your FINAL DIAGNOSIS and provide a 2-sentence clinical justification based strictly on the symptoms mentioned.
    HINT: Osteoarthritis often presents without the redness and warmth typical of Rheumatoid Arthritis.
    
    HISTORY:
    {full_history} [/INST]"""
    
    inputs = doctor_tok(final_prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        ids = doctor_model.generate(
            **inputs, 
            max_new_tokens=300, 
            temperature=0.1, 
            do_sample=True, 
            pad_token_id=doctor_tok.eos_token_id
        )
    
    final_diagnosis = doctor_tok.decode(ids[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    
    print(f"\n🩺 [FINAL DIAGNOSIS]: {final_diagnosis}")
    transcript.append({"role": "Final Diagnosis", "content": final_diagnosis})

    # --- SAVE RESULTS ---
    output_data = {
        "pathology_target": PATHOLOGY,
        "patient_model": PATIENT_MODEL_ID,
        "doctor_model": DOCTOR_MODEL_ID,
        "transcript": transcript
    }
    
    output_path = "../results/biomistral/session_transcript.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\n✅ Simulation complete. Results saved to {output_path}")

    # --- PHASE 3: VRAM CLEANUP ---
    print("🧹 Sweeping GPU memory to prepare for the Judge script...")
    
    # Safely delete variables based on how they were loaded
    if PATIENT_MODEL_ID == DOCTOR_MODEL_ID:
        del shared_model
    else:
        del patient_model
        del doctor_model
        
    gc.collect()
    torch.cuda.empty_cache()
    print("✨ VRAM cleared successfully.")

if __name__ == "__main__":
    run_simulation()