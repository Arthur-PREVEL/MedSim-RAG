import os
import sys
import glob
import json
import time
import re

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Prevent VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.orchestrator import MedSimOrchestrator

# ==========================================
# 1. MODEL REGISTRY (The core of the factory)
# ==========================================
MODELS_TO_BENCHMARK = {
    "llama3": {
        "hf_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "output_dir": "../results/benchmark/llama3/"
    },
    "mistral": {
        "hf_path": "mistralai/Mistral-7B-Instruct-v0.3",
        "output_dir": "../results/benchmark/mistral/"
    },
    "biomistral": {
        "hf_path": "BioMistral/BioMistral-7B-DARE",
        "output_dir": "../results/benchmark/biomistral/"
    },
    "gemma2": {
        "hf_path": "google/gemma-2-9b-it",
        "output_dir": "../results/benchmark/gemma2/"
    },
    "phi3": {
        "hf_path": "microsoft/Phi-3-mini-4k-instruct",
        "output_dir": "../results/benchmark/phi3/"
    }
}

# ==========================================
# GENERAL CONFIGURATION
# ==========================================
INPUT_DIR = "../results/biomistral/"  # Where we read the original transcripts from
DATA_PATH = "../data/knowledge_base_extract.json"

# ==========================================
# SYSTEM PROMPT (JSON Format Forcing)
# ==========================================
SYSTEM_PROMPT = """You are a strict, highly analytical Medical Professor evaluating a clinical student.
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

def extract_json_from_text(text):
    """
    Crucial function to clean the model's output.
    Robust version: auto-closes missing brackets if the LLM stopped prematurely.
    """
    text = text.strip()
    
    # Auto-repair: If the model forgot the final closing bracket
    if text.startswith('{') and not text.endswith('}'):
        text += '\n}'
        
    try:
        # Search for everything between the first { and the last }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            return None
    except json.JSONDecodeError:
        return None

def run_evaluation_for_model(model_key, config):
    print(f"\n{'='*80}")
    print(f"🚀 STARTING BENCHMARK FOR: {model_key.upper()} ({config['hf_path']})")
    print(f"{'='*80}\n")
    
    orchestrator = MedSimOrchestrator(file_path=DATA_PATH)
    
    # Check for input files
    transcript_files = glob.glob(os.path.join(INPUT_DIR, "transcript_*.json"))
    if not transcript_files:
        print(f"❌ No transcript files found in {INPUT_DIR}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)

    # VRAM Configuration (4-bit quantization)
    print(f"⌛ Loading model {model_key} into VRAM...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config["hf_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        config["hf_path"], 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    print(f"✅ {model_key.upper()} is ready.\n")

    # Evaluation Loop
    for index, input_file_path in enumerate(transcript_files, 1):
        filename = os.path.basename(input_file_path)
        output_file_path = os.path.join(config["output_dir"], filename)
        
        with open(input_file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            
        # 🧹 DATA CLEANSING: Remove legacy evaluations to keep the benchmark pure
        keys_to_remove = ["judge_evaluation", "llama3_evaluation"]
        for old_key in keys_to_remove:
            if old_key in data:
                del data[old_key]
            
        pathology = data.get("pathology_target", "Unknown")
        print("-" * 60)
        print(f"📄 Case {index}/{len(transcript_files)} : {pathology.upper()}")
        
        # Skip if already evaluated
        if os.path.exists(output_file_path):
            print("⏭️ Already evaluated. Skipping to next.")
            continue

        # Reconstruct the interview history
        interview_history = "\n".join([f"{m['role']}: {m['content']}" for m in data['transcript'] if m['role'] != 'Final Diagnosis'])
        final_diagnosis_given = next((m['content'] for m in data['transcript'] if m['role'] == 'Final Diagnosis'), "No diagnosis provided.")
        ground_truth = orchestrator.get_context(pathology, "judge")
        
        user_content = f"""### CLINICAL TRUTH:
{ground_truth}

### INTERVIEW TRANSCRIPT:
{interview_history}

### STUDENT'S FINAL DIAGNOSIS:
{final_diagnosis_given}

Please provide your evaluation now using the requested JSON format."""

        # Combine SYSTEM and USER prompts into a single USER message
        # This prevents template errors on older models (like BioMistral) 
        # that do not natively support the "system" role.
        combined_content = f"{SYSTEM_PROMPT}\n\n{user_content}"
        
        messages = [
            {"role": "user", "content": combined_content}
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
                max_new_tokens=600, # Increased because JSON takes more tokens
                temperature=0.1,    # Very low temperature to force strict structure
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        raw_evaluation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # 🌟 JSON EXTRACTION 🌟
        json_evaluation = extract_json_from_text(raw_evaluation)
        
        if json_evaluation:
            print(f"✅ JSON successfully generated! Total grade: {json_evaluation.get('total_grade_out_of_20', '?')}/20")
            data[f"benchmark_{model_key}"] = json_evaluation
        else:
            print("⚠️ JSON parsing failed. Saving raw text for debugging.")
            data[f"benchmark_{model_key}"] = {"raw_text_error": raw_evaluation}
            
        with open(output_file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        time.sleep(1)

    # Final VRAM Cleanup for the current model
    print(f"\n🧹 Sweeping VRAM for {model_key}...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    print("🏭 STARTING THE EVALUATION FACTORY (BENCHMARKING V1)")
    
    # Iterate over each model defined in the registry
    for model_key, config in MODELS_TO_BENCHMARK.items():
        run_evaluation_for_model(model_key, config)
        
    print("\n🎉 GLOBAL BENCHMARKING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
