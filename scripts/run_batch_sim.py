import os
import sys
import json
import time

# Add the parent directory to the import path for src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Preventing memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from orchestrator import MedSimOrchestrator

# ==========================================
# RESEARCH VARIABLES
# ==========================================
MODEL_ID = "BioMistral/BioMistral-7B-DARE"
TURNS = 4
DATA_PATH = '../data/knowledge_base_extract.json'
RESULTS_DIR = '../results/biomistral/'

def load_shared_agent(model_id):
    print(f"\n⚙️ INITIALISATION : Model loading {model_id} in VRAM...")
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
    print("✅ Model successfully loaded. ready for the batch !\n")
    return model, tokenizer

def run_batch():
    orchestrator = MedSimOrchestrator(file_path=DATA_PATH)
    
    #1. Retrieving the list of the 10 diseases
    pathologies = [item['title'] for item in orchestrator.db]
    print(f"📋 Batch processing has begun for {len(pathologies)} pathologies.")
    
    #2. Single model load (VRAM optimization)
    shared_model, shared_tok = load_shared_agent(MODEL_ID)
    patient_model, patient_tok = shared_model, shared_tok
    doctor_model, doctor_tok = shared_model, shared_tok

    # 3. Main Loop
    for index, pathology in enumerate(pathologies, 1):
        print("=" * 60)
        print(f"🚀 SIMULATION {index}/{len(pathologies)} : {pathology.upper()}")
        print("=" * 60)
        
        patient_context = orchestrator.get_context(pathology, "patient")
        transcript = []
        
        current_message = "Hello, I am the physician attending to you today. What symptoms are you experiencing?"
        transcript.append({"role": "Doctor", "content": current_message})
        print(f"🩺 [Doctor]: {current_message}")

        # --- PHASE 1: THE INTERVIEW ---
        for i in range(TURNS):
            # PATIENT TURN
            patient_prompt = f"""[INST] You are a 70-year-old patient. 
            SYMPTOMS CONTEXT: {patient_context}
            RULES: 
            1. Use simple language.
            2. Describe your pain naturally based ONLY on the context.
            3. Do NOT repeat what the doctor says.
            4. ABSOLUTE RULE: You are the PATIENT, do NOT act as a doctor.
            QUESTION FROM DOCTOR: {current_message} [/INST]"""
            
            inputs = patient_tok(patient_prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                ids = patient_model.generate(**inputs, max_new_tokens=150, temperature=0.8, do_sample=True, pad_token_id=patient_tok.eos_token_id)
            
            current_message = patient_tok.decode(ids[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
            current_message = current_message.replace("Patient:", "").replace("Answer:", "").replace('"', '').strip()
            
            print(f"👴 [Patient]: {current_message}")
            transcript.append({"role": "Patient", "content": current_message})

            if i == TURNS - 1:
                break

            # DOCTOR TURN
            doc_prompt = f"""[INST] You are a Doctor conducting a diagnostic interview.
            CURRENT INFO FROM PATIENT: {current_message}
            GOAL: Ask ONE brief question to help differentiate the patient's condition.
            CRITICAL RULE: Output ONLY the question you want to ask. [/INST]"""
            
            inputs = doctor_tok(doc_prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                ids = doctor_model.generate(**inputs, max_new_tokens=120, temperature=0.3, do_sample=True, pad_token_id=doctor_tok.eos_token_id)
                
            current_message = doctor_tok.decode(ids[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
            current_message = current_message.replace("Doctor:", "").strip()
            
            print(f"🩺 [Doctor]: {current_message}")
            transcript.append({"role": "Doctor", "content": current_message})

        # --- PHASE 2: FINAL DIAGNOSIS ---
        print("\n🧐 [Formulation du diagnostic...]")
        full_history = "\n".join([f"{m['role']}: {m['content']}" for m in transcript])
        final_prompt = f"""[INST] Review the following medical interview history. 
        State your FINAL DIAGNOSIS and provide a 2-sentence clinical justification based strictly on the symptoms mentioned.
        HISTORY:
        {full_history} [/INST]"""
        
        inputs = doctor_tok(final_prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            ids = doctor_model.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=True, pad_token_id=doctor_tok.eos_token_id)
        
        final_diagnosis = doctor_tok.decode(ids[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        print(f"🩺 [FINAL DIAGNOSIS]: {final_diagnosis}\n")
        transcript.append({"role": "Final Diagnosis", "content": final_diagnosis})

        #  --- SAVE FILE (Unique name per disease) ---
        safe_filename = pathology.replace(" ", "_").replace("/", "_").lower()
        output_path = os.path.join(RESULTS_DIR, f"transcript_{safe_filename}.json")
        
        output_data = {
            "pathology_target": pathology,
            "patient_model": MODEL_ID,
            "doctor_model": MODEL_ID,
            "transcript": transcript
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)
        
        print(f"💾 Sauvegardé : {output_path}")
        time.sleep(1) # A quick break for the CPU

    # --- PHASE 3: FINAL VRAM CLEANING ---
    print("\n🧹 Batch processing complete. Cleaning VRAM...")
    del shared_model
    gc.collect()
    torch.cuda.empty_cache()
    print("✨ VRAM successfully clean")

if __name__ == "__main__":
    # Create the results folder if it does not exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_batch()
