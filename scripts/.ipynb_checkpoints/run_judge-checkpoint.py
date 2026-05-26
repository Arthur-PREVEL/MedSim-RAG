import os
import sys
import argparse
import logging
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.orchestrator import MedSimOrchestrator
import json

gc.collect()
torch.cuda.empty_cache()

# ==========================================
# MODEL REGISTRY
# ==========================================
MODELS = {
    "biomistral": "BioMistral/BioMistral-7B-DARE",
    "llama":      "meta-llama/Llama-3.1-8B-Instruct",
}

# ==========================================
# ARGUMENT PARSING
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="MedSim-RAG Judge Evaluation")
    parser.add_argument(
        "--judge",
        choices=MODELS.keys(),
        default="biomistral",
        help="Model to use as judge (default: biomistral)"
    )
    parser.add_argument(
        "--transcript",
        type=str,
        required=True,
        help="Path to the session transcript JSON file"
    )
    return parser.parse_args()

# ==========================================
# PROMPT BUILDERS
# ==========================================
def build_judge_prompt(judge_name, ground_truth, interview_history, final_diagnosis_given):
    task = """Evaluate the student on exactly these 4 criteria. End each criterion with "Score: X/5".
    1. Diagnosis Comparison (out of 5)
    2. Key Symptoms Comparison (out of 5)
    3. Interview Technique (out of 5)
    4. Clinical Justification (out of 5)"""

    if judge_name == "llama":
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"You are a strict Medical Professor. Output only the evaluation. No markdown, no bold, no preamble. No text before section 1.\n"
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"### CLINICAL TRUTH:\n{ground_truth}\n\n"
            f"### INTERVIEW TRANSCRIPT:\n{interview_history}\n\n"
            f"### STUDENT'S FINAL DIAGNOSIS:\n{final_diagnosis_given}\n\n"
            f"### TASK:\n"
            f"Write exactly 4 numbered sections. Each section ends with 'Score: X/5'.\n"
            f"Use this exact format:\n"
            f"1. Diagnosis Comparison: [feedback] Score: X/5\n"
            f"2. Key Symptoms Comparison: [feedback] Score: X/5\n"
            f"3. Interview Technique: [feedback] Score: X/5\n"
            f"4. Clinical Justification: [feedback] Score: X/5\n"
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    else:  # biomistral
        return (
            f"[INST] You are a strict Medical Professor evaluating a clinical student.\n\n"
            f"### CLINICAL TRUTH:\n{ground_truth}\n\n"
            f"### INTERVIEW TRANSCRIPT:\n{interview_history}\n\n"
            f"### STUDENT'S FINAL DIAGNOSIS:\n{final_diagnosis_given}\n\n"
            f"### TASK:\n{task}\n"
            f"[/INST]\n"
            f"### EVALUATION REPORT\n1. Diagnosis Comparison:"
        )

# ==========================================
# RESPONSE PARSING
# ==========================================
def parse_judge_response(judge_name, generated_text):
    if judge_name == "llama":
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            text = text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
        else:
            text = generated_text
        evaluation = text
    else:  # biomistral
        evaluation = "1. Diagnosis Comparison:" + generated_text.split("1. Diagnosis Comparison:")[-1].strip()

    # Tronquer après le point 5 seulement si un point 6 existe
    if re.search(r'\n?6\.', evaluation):
        evaluation = re.split(r'\n?6\.', evaluation)[0].strip()

    # Recalculer le total depuis les scores individuels
    scores = re.findall(r'Score:\s*(\d+)/5', evaluation)
    if len(scores) >= 4:
        total = sum(int(s) for s in scores[:4])
    else:
        # Fallback: chercher pattern X/5
        scores = re.findall(r'(\d+)/5', evaluation)
        total = sum(int(s) for s in scores[:4]) if len(scores) >= 4 else None

    if total is not None:
        evaluation = re.sub(r'5\.\s*Total Grade.*', f'5. Total Grade: {total}/20.', evaluation)
        if '5. Total Grade' not in evaluation:
            evaluation += f"\n5. Total Grade: {total}/20."

    # Formatage : retours à la ligne entre les critères
    evaluation = re.sub(r'(?m)^(\d+\.)\s', r'\n\1 ', evaluation).strip()

    return evaluation, total

# ==========================================
# MAIN EVALUATION
# ==========================================
def evaluate_session(judge_name, transcript_path):
    orchestrator = MedSimOrchestrator()

    # 1. Load transcript
    try:
        with open(transcript_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Transcript not found at {transcript_path}. Run simulation first!")
        return

    interview_history = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in data['transcript'] if m['role'] != 'Final Diagnosis'
    ])
    final_diagnosis_given = next(
        (m['content'] for m in data['transcript'] if m['role'] == 'Final Diagnosis'),
        "No diagnosis provided."
    )
    ground_truth = orchestrator.get_context(data['pathology_target'], "judge")

    # 2. Load judge model
    judge_model_id = MODELS[judge_name]
    print(f"⌛ Loading Judge Model: {judge_model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(judge_model_id, clean_up_tokenization_spaces=False)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 3. Build prompt and generate
    eval_prompt = build_judge_prompt(judge_name, ground_truth, interview_history, final_diagnosis_given)
    print("🧠 The Professor is analyzing the case...")
    inputs = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id if judge_name == "llama" else None
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=(judge_name != "llama"))

    # 4. Parse and display
    evaluation, total = parse_judge_response(judge_name, generated_text)

    print("\n" + "="*50)
    print("🎓 PROFESSOR EVALUATION")
    print("="*50)
    print(evaluation)
    print("="*50)

    # 5. Save evaluation alongside transcript
    transcript_dir = os.path.dirname(transcript_path)
    output_path = os.path.join(transcript_dir, f"evaluation_judge_{judge_name}.json")
    eval_data = {
        "pathology_target": data['pathology_target'],
        "patient_model": data['patient_model'],
        "doctor_model": data['doctor_model'],
        "judge_model": judge_model_id,
        "total_grade": total,
        "evaluation_text": evaluation
    }
    with open(output_path, "w") as f:
        json.dump(eval_data, f, indent=4)
    print(f"\n✅ Evaluation saved to {output_path}")

    # 6. VRAM cleanup
    print("\n🧹 Sweeping GPU memory...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("✨ Judge finished. VRAM cleared successfully.")

if __name__ == "__main__":
    args = parse_args()
    print(f"🔧 Judge: {args.judge} | Transcript: {args.transcript}")
    evaluate_session(args.judge, args.transcript)