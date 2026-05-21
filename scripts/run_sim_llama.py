import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.orchestrator import MedSimOrchestrator
import json

# ==========================================
# RESEARCH VARIABLES
# ==========================================
PATIENT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"  # Patient: Llama 3.1
DOCTOR_MODEL_ID = "BioMistral/BioMistral-7B-DARE"        # Doctor: BioMistral (unchanged)
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
    tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model, tokenizer

def build_llama_patient_prompt(patient_context, doctor_message):
    """Build a Llama 3.1 Instruct formatted prompt for the patient agent."""
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

def parse_llama_response(decoded_text):
    """Extract only the assistant's response from Llama 3.1 output."""
    if "<|start_header_id|>assistant<|end_header_id|>" in decoded_text:
        response = decoded_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    else:
        response = decoded_text
    # Clean up any residual special tokens or artifacts
    response = response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
    response = response.replace("Jean:", "").replace("Patient:", "").replace("Answer:", "").strip()
    return response

def run_simulation():
    orchestrator = MedSimOrchestrator(file_path='../data/knowledge_base_extract.json')

    # Patient and Doctor are different models — load both
    print("💡 Loading Patient (Llama 3.1 8B) and Doctor (BioMistral) separately...")
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

        # --- PATIENT TURN (Llama 3.1) ---
        patient_prompt = build_llama_patient_prompt(patient_context, current_message)

        inputs = patient_tok(patient_prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            ids = patient_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                do_sample=True,
                pad_token_id=patient_tok.eos_token_id
            )

        current_message = parse_llama_response(patient_tok.decode(ids[0], skip_special_tokens=False))
        print(f"👴 [Patient]: {current_message}")
        transcript.append({"role": "Patient", "content": current_message})

        # Stop after last patient turn
        if i == TURNS - 1:
            break

        # --- DOCTOR TURN (BioMistral — format unchanged) ---
        doc_prompt = f"""[INST] You are a Doctor conducting a diagnostic interview.
        CURRENT INFO FROM PATIENT: {current_message}
        GOAL: Ask ONE brief question to help differentiate the patient's condition. Focus on ruling out other forms of arthritis.
        CRITICAL RULE: Output ONLY the question you want to ask. Do not write your internal thoughts. [/INST]"""

        inputs = doctor_tok(doc_prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            ids = doctor_model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.3,
                do_sample=True,
                pad_token_id=doctor_tok.eos_token_id
            )

        current_message = doctor_tok.decode(ids[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        current_message = current_message.replace("Doctor:", "").strip()

        print(f"🩺 [Doctor]: {current_message}")
        transcript.append({"role": "Doctor", "content": current_message})

    # --- PHASE 2: FINAL DIAGNOSIS (BioMistral) ---
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

    output_path = "../results/llama/session_transcript.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\n✅ Simulation complete. Results saved to {output_path}")

    # --- VRAM CLEANUP ---
    print("🧹 Sweeping GPU memory...")
    del patient_model
    del doctor_model
    gc.collect()
    torch.cuda.empty_cache()
    print("✨ VRAM cleared successfully.")

if __name__ == "__main__":
    run_simulation()