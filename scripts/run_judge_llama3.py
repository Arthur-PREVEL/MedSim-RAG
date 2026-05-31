import os
import sys
import glob
import json
import time

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Prevent VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.orchestrator import MedSimOrchestrator

# ==========================================
# CONFIGURATION
# ==========================================
JUDGE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
INPUT_DIR = "../results/biomistral/"  # Where we read the transcripts from
OUTPUT_DIR = "../results/llama3/"     # Where we save the Llama-3 graded files
DATA_PATH = "../data/knowledge_base_extract.json"

def evaluate_batch():
    orchestrator = MedSimOrchestrator(file_path=DATA_PATH)
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Find all transcripts
    transcript_files = glob.glob(os.path.join(INPUT_DIR, "transcript_*.json"))
    if not transcript_files:
        print(f"❌ No transcript files found in {INPUT_DIR}")
        return

    print(f"📋 Starting Llama-3 Batch Evaluation: {len(transcript_files)} files to grade.")

    # 2. Load Llama-3 with VRAM Optimizations (4-bit)
    print(f"\n⌛ Loading Professor Model ({JUDGE_MODEL})...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    print("✅ Llama-3 Professor is ready.\n")

    # 3. Evaluation Loop
    for index, input_file_path in enumerate(transcript_files, 1):
        filename = os.path.basename(input_file_path)
        output_file_path = os.path.join(OUTPUT_DIR, filename)
        
        with open(input_file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            
        pathology = data.get("pathology_target", "Unknown")
        print("=" * 60)
        print(f"🎓 LLAMA-3 GRADING {index}/{len(transcript_files)} : {pathology.upper()}")
        print("=" * 60)
        
        # Check if the file already exists in the Llama-3 folder
        if os.path.exists(output_file_path):
            print("⏭️ File already exists in Llama-3 folder. Skipping.")
            continue

        # Reconstruct the interview history
        interview_history = "\n".join([f"{m['role']}: {m['content']}" for m in data['transcript'] if m['role'] != 'Final Diagnosis'])
        final_diagnosis_given = next((m['content'] for m in data['transcript'] if m['role'] == 'Final Diagnosis'), "No diagnosis provided.")
        ground_truth = orchestrator.get_context(pathology, "judge")

        # --- PROMPT ENGINEERING FOR LLAMA-3 ---
        system_instruction = """You are a strict and highly analytical Medical Professor evaluating a clinical student.
Your task is to compare the student's interview and final diagnosis against the CLINICAL TRUTH.

CRITICAL RULE: You MUST output a structured evaluation and conclude with a final grade out of 20 (e.g., 'Overall Grade: 14/20').
You MUST begin your response EXACTLY with the phrase '1. Diagnosis Comparison:'."""
        
        user_content = f"""### CLINICAL TRUTH:
{ground_truth}

### INTERVIEW TRANSCRIPT:
{interview_history}

### STUDENT'S FINAL DIAGNOSIS:
{final_diagnosis_given}

Please provide your evaluation now."""

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]
        
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=400, 
                temperature=0.2, 
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        evaluation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        print("\n📝 PROFESSOR'S EVALUATION:")
        print(evaluation)
        print("-" * 60)
        
        # Update the dictionary and save to the NEW folder
        data["llama3_evaluation"] = evaluation
        with open(output_file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        print(f"\n💾 Llama-3 grade saved to {output_file_path}.")
        time.sleep(1)

    print("\n🧹 Batch evaluation complete. Sweeping VRAM...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("✨ VRAM cleared successfully.")

if __name__ == "__main__":
    evaluate_batch()
