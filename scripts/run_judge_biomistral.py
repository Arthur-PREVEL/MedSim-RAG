import os
import sys

# UPDATED: Add the parent directory to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Must be set before importing torch to prevent memory fragmentation OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.orchestrator import MedSimOrchestrator
import json

# --- INITIAL VRAM CLEANUP ---
gc.collect()
torch.cuda.empty_cache()

JUDGE_MODEL = "BioMistral/BioMistral-7B-DARE"

# UPDATED: Point to the new results directory
TRANSCRIPT_PATH = "../results/biomistral/session_transcript.json"

def evaluate_session():
    orchestrator = MedSimOrchestrator()
    
    # 1. Load the Transcript
    try:
        with open(TRANSCRIPT_PATH, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Transcript not found at {TRANSCRIPT_PATH}. Run simulation first!")
        return
    
    interview_history = "\n".join([f"{m['role']}: {m['content']}" for m in data['transcript'] if m['role'] != 'Final Diagnosis'])
    final_diagnosis_given = next((m['content'] for m in data['transcript'] if m['role'] == 'Final Diagnosis'), "No diagnosis provided.")
    
    ground_truth = orchestrator.get_context(data['pathology_target'], "judge")

    # 2. Load the Judge Model
    print(f"⌛ Loading Judge Model: {JUDGE_MODEL}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, 
        quantization_config=bnb_config, 
        device_map="auto"
    )

    # 3. Create Assessment Prompt (With Forced Scaffolding)
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
1. Diagnosis Comparison:""" # <-- The scaffolding hangs here!

    print("🧠 The Professor is analyzing the case...")
    inputs = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=400, 
            temperature=0.3, # Slightly higher to encourage more detailed feedback
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 4. Clean up and format the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # We re-attach the scaffold string so the final output reads cleanly
    evaluation = "1. Diagnosis Comparison:" + generated_text.split("1. Diagnosis Comparison:")[-1].strip()
    
    print("\n" + "="*50)
    print("🎓 PROFESSOR EVALUATION")
    print("="*50)
    print(evaluation)
    print("="*50)

    # --- FINAL VRAM CLEANUP ---
    print("\n🧹 Sweeping GPU memory...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("✨ Judge finished. VRAM cleared successfully.")

if __name__ == "__main__":
    evaluate_session()
