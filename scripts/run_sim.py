import os
import sys
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.orchestrator import MedSimOrchestrator
import json

# ==========================================
# MODEL REGISTRY
# ==========================================
MODELS = {
    "biomistral": "BioMistral/BioMistral-7B-DARE",
    "llama":      "meta-llama/Llama-3.1-8B-Instruct",
}

PATHOLOGY = "Osteoarthritis"
TURNS = 4

# ==========================================
# ARGUMENT PARSING
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="MedSim-RAG Dual-Agent Simulation")
    parser.add_argument(
        "--patient",
        choices=MODELS.keys(),
        default="biomistral",
        help="Model to use for the patient agent (default: biomistral)"
    )
    parser.add_argument(
        "--doctor",
        choices=MODELS.keys(),
        default="biomistral",
        help="Model to use for the doctor agent (default: biomistral)"
    )
    return parser.parse_args()

# ==========================================
# MODEL LOADING
# ==========================================
def load_agent(model_id):
    print(f"⌛ Loading model: {model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model, tokenizer

# ==========================================
# PROMPT BUILDERS
# ==========================================
def build_patient_prompt(model_name, patient_context, doctor_message):
    if model_name == "llama":
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"You are Jean, a 70-year-old patient visiting a doctor.\n"
            f"SYMPTOMS CONTEXT: {patient_context}\n"
            f"RULES:\n"
            f"1. Use simple, everyday language. No medical terms like 'nodes', 'locomotor', or 'degenerative'.\n"
            f"2. Describe your pain and discomfort naturally, as a real elderly person would.\n"
            f"3. Do NOT repeat what the doctor says.\n"
            f"4. Do NOT act as a doctor or provide diagnoses.\n"
            f"5. Stay in character as Jean at all times. No lists, no numbering.\n"
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{doctor_message}"
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    else:  # biomistral
        return (
            f"[INST] You are Jean, a 70-year-old patient.\n"
            f"SYMPTOMS CONTEXT: {patient_context}\n"
            f"RULES:\n"
            f"1. Use simple language (no medical terms like 'nodes' or 'locomotor').\n"
            f"2. Describe your pain naturally.\n"
            f"3. Do NOT repeat what the doctor says.\n"
            f"4. ABSOLUTE RULE: You are the PATIENT, do NOT act as a doctor.\n"
            f"QUESTION FROM DOCTOR: {doctor_message} [/INST]"
        )

def build_doctor_prompt(model_name, patient_message):
    if model_name == "llama":
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"You are a Doctor conducting a diagnostic interview.\n"
            f"GOAL: Ask ONE brief question to help differentiate the patient's condition. Focus on ruling out other forms of arthritis.\n"
            f"CRITICAL RULE: Output ONLY the question. No internal thoughts, no preamble.\n"
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Patient said: {patient_message}"
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    else:  # biomistral
        return (
            f"[INST] You are a Doctor conducting a diagnostic interview.\n"
            f"CURRENT INFO FROM PATIENT: {patient_message}\n"
            f"GOAL: Ask ONE brief question to help differentiate the patient's condition. Focus on ruling out other forms of arthritis.\n"
            f"CRITICAL RULE: Output ONLY the question you want to ask. Do not write your internal thoughts. [/INST]"
        )

def build_diagnosis_prompt(model_name, full_history):
    if model_name == "llama":
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"You are a Doctor formulating a final diagnosis.\n"
            f"Review the interview history and state your FINAL DIAGNOSIS with a 2-sentence clinical justification based strictly on the symptoms mentioned.\n"
            f"HINT: Osteoarthritis often presents without the redness and warmth typical of Rheumatoid Arthritis.\n"
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"HISTORY:\n{full_history}"
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    else:  # biomistral
        return (
            f"[INST] Review the following medical interview history.\n"
            f"State your FINAL DIAGNOSIS and provide a 2-sentence clinical justification based strictly on the symptoms mentioned.\n"
            f"HINT: Osteoarthritis often presents without the redness and warmth typical of Rheumatoid Arthritis.\n\n"
            f"HISTORY:\n{full_history} [/INST]"
        )

# ==========================================
# RESPONSE PARSERS
# ==========================================
def parse_response(model_name, decoded_text):
    if model_name == "llama":
        if "<|start_header_id|>assistant<|end_header_id|>" in decoded_text:
            response = decoded_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        else:
            response = decoded_text
        response = response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
    else:  # biomistral
        response = decoded_text.split("[/INST]")[-1].strip()

    # Clean common artifacts
    response = response.replace("Patient:", "").replace("Jean:", "").replace("Answer:", "")
    response = response.replace("Doctor:", "").strip()
    return response

# ==========================================
# MAIN SIMULATION
# ==========================================
def run_simulation(patient_name, doctor_name):
    patient_model_id = MODELS[patient_name]
    doctor_model_id = MODELS[doctor_name]

    orchestrator = MedSimOrchestrator(file_path='../data/knowledge_base_extract.json')

    # VRAM optimization: load once if same model
    if patient_model_id == doctor_model_id:
        print(f"💡 VRAM Optimization: same model for Patient & Doctor. Loading once.")
        shared_model, shared_tok = load_agent(patient_model_id)
        patient_model, patient_tok = shared_model, shared_tok
        doctor_model, doctor_tok = shared_model, shared_tok
    else:
        print(f"💡 Loading Patient ({patient_name}) and Doctor ({doctor_name}) separately...")
        patient_model, patient_tok = load_agent(patient_model_id)
        doctor_model, doctor_tok = load_agent(doctor_model_id)

    patient_context = orchestrator.get_context(PATHOLOGY, "patient")
    transcript = []

    print(f"\n🚀 DUAL-AGENT SIMULATION: {PATHOLOGY.upper()}")
    print(f"   Patient: {patient_name.upper()} | Doctor: {doctor_name.upper()}")
    print("-" * 50)

    current_message = "Hello, I am the physician attending to you today. What symptoms are you experiencing?"
    transcript.append({"role": "Doctor", "content": current_message})
    print(f"🩺 [Doctor]: {current_message}")

    # --- PHASE 1: CLINICAL INTERVIEW ---
    for i in range(TURNS):

        # --- PATIENT TURN ---
        patient_prompt = build_patient_prompt(patient_name, patient_context, current_message)
        inputs = patient_tok(patient_prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            ids = patient_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                do_sample=True,
                pad_token_id=patient_tok.eos_token_id
            )
        decoded = patient_tok.decode(ids[0], skip_special_tokens=(patient_name != "llama"))
        current_message = parse_response(patient_name, decoded)
        print(f"👴 [Patient]: {current_message}")
        transcript.append({"role": "Patient", "content": current_message})

        if i == TURNS - 1:
            break

        # --- DOCTOR TURN ---
        doc_prompt = build_doctor_prompt(doctor_name, current_message)
        inputs = doctor_tok(doc_prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            ids = doctor_model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.3,
                do_sample=True,
                pad_token_id=doctor_tok.eos_token_id
            )
        decoded = doctor_tok.decode(ids[0], skip_special_tokens=(doctor_name != "llama"))
        current_message = parse_response(doctor_name, decoded)
        print(f"🩺 [Doctor]: {current_message}")
        transcript.append({"role": "Doctor", "content": current_message})

    # --- PHASE 2: FINAL DIAGNOSIS ---
    print("\n🧐 [Doctor is formulating the final diagnosis...]")
    full_history = "\n".join([f"{m['role']}: {m['content']}" for m in transcript])
    final_prompt = build_diagnosis_prompt(doctor_name, full_history)

    inputs = doctor_tok(final_prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        ids = doctor_model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
            pad_token_id=doctor_tok.eos_token_id
        )
    decoded = doctor_tok.decode(ids[0], skip_special_tokens=(doctor_name != "llama"))
    final_diagnosis = parse_response(doctor_name, decoded)

    print(f"\n🩺 [FINAL DIAGNOSIS]: {final_diagnosis}")
    transcript.append({"role": "Final Diagnosis", "content": final_diagnosis})

    # --- SAVE RESULTS ---
    output_data = {
        "pathology_target": PATHOLOGY,
        "patient_model": patient_model_id,
        "doctor_model": doctor_model_id,
        "transcript": transcript
    }

    output_dir = f"../results/patient_{patient_name}_doctor_{doctor_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "session_transcript.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\n✅ Simulation complete. Results saved to {output_path}")

    # --- VRAM CLEANUP ---
    print("🧹 Sweeping GPU memory...")
    if patient_model_id == doctor_model_id:
        del shared_model
    else:
        del patient_model
        del doctor_model
    gc.collect()
    torch.cuda.empty_cache()
    print("✨ VRAM cleared successfully.")

if __name__ == "__main__":
    args = parse_args()
    print(f"🔧 Configuration — Patient: {args.patient} | Doctor: {args.doctor}")
    run_simulation(args.patient, args.doctor)