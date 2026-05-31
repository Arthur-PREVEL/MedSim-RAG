import os
import sys
import glob
import json
import time

# Add the parent folder to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from orchestrator import MedSimOrchestrator

# ==========================================
# VARIABLES
# ==========================================
JUDGE_MODEL = "BioMistral/BioMistral-7B-DARE"
RESULTS_DIR = "../results/biomistral/"
DATA_PATH = "../data/knowledge_base_extract.json"

def evaluate_batch():
    orchestrator = MedSimOrchestrator(file_path=DATA_PATH)
    
    # 1. Find all generated transcripts
    transcript_files = glob.glob(os.path.join(RESULTS_DIR, "transcript_*.json"))
    if not transcript_files:
        print(f"❌ No transcript found dans {RESULTS_DIR}")
        return

    print(f"📋 Start of batch evaluation : {len(transcript_files)} papers to grade.")

    # 2. Judge loading
    print(f"\n⌛ Loading the Professor ({JUDGE_MODEL})...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    print("✅ Professor ready.\n")

    # 3. Correction loop
    for index, file_path in enumerate(transcript_files, 1):
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            
        pathology = data.get("pathology_target", "Unknown")
        print("=" * 60)
        print(f"🎓 CORRECTION {index}/{len(transcript_files)} : {pathology.upper()}")
        print("=" * 60)
        
        # If it has already been corrected, we move on
        if "judge_evaluation" in data:
            print("⏭️ Already rated. Move on to the next one.")
            continue

        interview_history = "\n".join([f"{m['role']}: {m['content']}" for m in data['transcript'] if m['role'] != 'Final Diagnosis'])
        final_diagnosis_given = next((m['content'] for m in data['transcript'] if m['role'] == 'Final Diagnosis'), "No diagnosis provided.")
        ground_truth = orchestrator.get_context(pathology, "judge")

        eval_prompt = f"""[INST] You are a strict Medical Professor evaluating a clinical student.
        
        ### CLINICAL TRUTH:
        {ground_truth}
        
        ### INTERVIEW TRANSCRIPT:
        {interview_history}
        
        ### STUDENT'S FINAL DIAGNOSIS:
        {final_diagnosis_given}
        
        ### TASK:
        Evaluate the student's performance based on the transcript and the clinical truth. Provide a grade out of 20 and specific feedback.
        [/INST]
### EVALUATION REPORT
1. Diagnosis Comparison:""" # <-- Scaffolding trick

        inputs = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=400, 
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        evaluation = "1. Diagnosis Comparison:" + generated_text.split("1. Diagnosis Comparison:")[-1].strip()
        
        print(evaluation)
        
        # 4. Save the score to the JSON file
        data["judge_evaluation"] = evaluation
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        print(f"\n💾 Note saved in the JSON file.")
        time.sleep(1)

    # --- NETTOYAGE VRAM ---
    print("\n🧹 All copies have been graded. Clearing VRAM...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("✨ VRAM successfully freed.")

if __name__ == "__main__":
    evaluate_batch()
