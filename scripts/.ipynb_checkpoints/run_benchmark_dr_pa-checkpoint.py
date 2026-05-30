import os
import sys
import argparse
import logging
import json
import gc
import itertools
import time
import re
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.orchestrator import MedSimOrchestrator

# ==========================================
# 1. MODEL REGISTRY
# ==========================================
MODELS = {
    "biomistral": {
        "hf_path":    "BioMistral/BioMistral-7B-DARE",
        "output_dir": "../results/benchmark_simulation/biomistral/",
    },
    "llama3": {
        "hf_path":    "meta-llama/Llama-3.1-8B-Instruct",
        "output_dir": "../results/benchmark_simulation/llama3/",
    },
    "mistral": {
        "hf_path":    "mistralai/Mistral-7B-Instruct-v0.3",
        "output_dir": "../results/benchmark_simulation/mistral/",
    },
    "gemma2": {
        "hf_path":    "google/gemma-2-9b-it",
        "output_dir": "../results/benchmark_simulation/gemma2/",
    },
    "phi3": {
        "hf_path":    "microsoft/Phi-3-mini-4k-instruct",
        "output_dir": "../results/benchmark_simulation/phi3/",
    },
}

# ==========================================
# 2. GENERAL CONFIGURATION
# ==========================================
JUDGE_MODEL_KEY = "llama3"
ALL_CONFIGS     = list(itertools.product(MODELS.keys(), MODELS.keys()))

TURNS     = 4
DATA_PATH = "../data/knowledge_base_extract.json"

# ==========================================
# 3. LLM JUDGE — SYSTEM PROMPTS
# ==========================================
JUDGE_SYSTEM_PATIENT = """You are a strict medical simulation evaluator assessing the quality of an AI-simulated patient.
Your task is to evaluate whether the patient's responses are realistic, natural, and free of medical jargon.

CRITICAL INSTRUCTION: Output STRICTLY a valid JSON object. No markdown, no preamble, no text outside the JSON.

Use exactly this JSON format:
{
    "natural_language": {
        "score_out_of_4": 0,
        "justification": "Did the patient use everyday language, avoiding medical terms?"
    },
    "symptom_coherence": {
        "score_out_of_3": 0,
        "justification": "Were the symptoms described consistent with the clinical context provided?"
    },
    "role_adherence": {
        "score_out_of_3": 0,
        "justification": "Did the patient stay in character and avoid acting like a doctor?"
    },
    "total_score_out_of_10": 0,
    "overall_feedback": "A concise one-paragraph summary of the patient simulation quality."
}"""

JUDGE_SYSTEM_DOCTOR = """You are a strict medical simulation evaluator assessing the quality of an AI-simulated doctor.
Your task is to evaluate whether the doctor asked clinically relevant questions and produced a sound final diagnosis.

CRITICAL INSTRUCTION: Output STRICTLY a valid JSON object. No markdown, no preamble, no text outside the JSON.

Use exactly this JSON format:
{
    "question_quality": {
        "score_out_of_4": 0,
        "justification": "Were the questions targeted, clinically relevant, and appropriate for differential diagnosis?"
    },
    "diagnostic_accuracy": {
        "score_out_of_3": 0,
        "justification": "Was the final diagnosis correct or reasonable given the symptoms and clinical truth?"
    },
    "role_adherence": {
        "score_out_of_3": 0,
        "justification": "Did the doctor stay professional and avoid acting like a patient?"
    },
    "total_score_out_of_10": 0,
    "overall_feedback": "A concise one-paragraph summary of the doctor simulation quality."
}"""

JARGON_SYSTEM_PROMPT = """You are a medical terminology expert.
Your task is to list medical and clinical terms that a REAL PATIENT would NOT use in everyday speech,
but that an AI simulating a patient might incorrectly produce.

CRITICAL INSTRUCTION: Output STRICTLY a valid JSON object. No markdown, no preamble.

Use exactly this format:
{
    "jargon_terms": ["term1", "term2", "term3", ...]
}

Rules:
- Include 15 to 25 terms.
- All terms in lowercase.
- Focus on terms specific to the given pathology AND general medical jargon a layperson would avoid.
- Include both full terms and common abbreviations (e.g. "oa", "ra", "nsaid").
"""

# Fallback list used when --no-judge or when jargon generation fails
MEDICAL_JARGON_FALLBACK = [
    "degenerative", "locomotor", "nodes", "cartilage", "synovial",
    "inflammation", "pathology", "diagnosis", "prognosis", "etiology",
    "rheumatoid", "osteoarthritis", "arthritis", "condition", "treatment",
    "medication", "prescription", "clinical", "symptomatology", "chronic"
]

DOCTOR_ROLE_LEAKAGE = [
    "i diagnose", "my diagnosis", "treatment would be", "i prescribe",
    "you have", "you are suffering from", "based on your symptoms, i believe you have"
]

PATIENT_ROLE_LEAKAGE = [
    "as a doctor", "my patient", "i recommend", "i suggest", "i prescribe",
    "you should take", "the treatment is", "medically speaking"
]

# ==========================================
# 4. JSON EXTRACTOR
# ==========================================
def extract_json_from_text(text: str):
    text = text.strip()
    if text.startswith('{') and not text.endswith('}'):
        text += '\n}'
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except json.JSONDecodeError:
        return None

# ==========================================
# 5. ARGUMENT PARSING
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="MedSim-RAG Dual-Agent Benchmark — Patient × Doctor combinations",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--configs", type=str, default="all",
        help=(
            "Which configurations to run. Examples:\n"
            "  --configs all          → all 25 combinations\n"
            "  --configs 1-5          → configs 1 to 5\n"
            "  --configs 1,3,7,12     → specific configs\n"
            "  --configs 1-5,10,15    → mixed range and individual\n\n"
            "Config index list (patient_model x doctor_model):\n" +
            "\n".join([f"  {i+1:2d}. patient={p.upper():12s} | doctor={d.upper()}"
                       for i, (p, d) in enumerate(ALL_CONFIGS)])
        )
    )
    parser.add_argument("--list",     action="store_true",
                        help="List all available configurations and exit")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM judge step (faster, heuristic scores only)")
    return parser.parse_args()


def parse_config_selection(selection: str) -> list:
    indices = set()
    for part in selection.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            for i in range(int(start), int(end) + 1):
                indices.add(i - 1)
        else:
            indices.add(int(part) - 1)
    return sorted(indices)

# ==========================================
# 6. MODEL LOADING / UNLOADING
# ==========================================
def load_model(model_key: str):
    model_id = MODELS[model_key]["hf_path"]
    free, total = torch.cuda.mem_get_info()
    print(f"   VRAM free: {free/1e9:.1f}GB / {total/1e9:.1f}GB")
    print(f"⌛ Loading {model_key.upper()} ({model_id})...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_key in ("gemma2", "phi3"):
        # Répartition sur les 2 GPUs — GPU 1 réservé pour overflow
        print(f"   Dual-GPU mode activated for {model_key.upper()}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={
                0: "4GiB",   # limite stricte sur GPU 0 (déjà occupé par patient+judge)
                1: "14GiB",  # GPU 1 quasi exclusif pour Gemma2/Phi3
                "cpu": "48GiB"
            }
        )
    else:
        # BioMistral, Llama3, Mistral → GPU 0 uniquement, limité à 7GiB
        # pour laisser de la place à l'inférence
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={
                0: "7GiB",
                "cpu": "48GiB"
            }
        )

    print(f"✅ {model_key.upper()} ready.")
    return model, tokenizer


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(10)

# ==========================================
# 7. PROMPT BUILDERS — SIMULATION
# ==========================================
def build_patient_prompt(model_key: str, patient_context: str, doctor_message: str) -> str:
    if model_key == "llama3":
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
    elif model_key == "gemma2":
        return (
            f"<start_of_turn>user\n"
            f"You are Jean, a 70-year-old patient visiting a doctor.\n"
            f"SYMPTOMS CONTEXT: {patient_context}\n"
            f"RULES:\n"
            f"1. Use simple language (no medical terms like 'nodes' or 'locomotor').\n"
            f"2. Describe your pain naturally.\n"
            f"3. Do NOT repeat what the doctor says.\n"
            f"4. ABSOLUTE RULE: You are the PATIENT, do NOT act as a doctor.\n"
            f"QUESTION FROM DOCTOR: {doctor_message}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    elif model_key == "phi3":
        return (
            f"<|user|>\n"
            f"You are Jean, a 70-year-old patient visiting a doctor.\n"
            f"SYMPTOMS CONTEXT: {patient_context}\n"
            f"RULES:\n"
            f"1. Use simple language (no medical terms like 'nodes' or 'locomotor').\n"
            f"2. Describe your pain naturally.\n"
            f"3. Do NOT repeat what the doctor says.\n"
            f"4. ABSOLUTE RULE: You are the PATIENT, do NOT act as a doctor.\n"
            f"QUESTION FROM DOCTOR: {doctor_message}<|end|>\n"
            f"<|assistant|>\n"
        )
    else:  # biomistral / mistral
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


def build_doctor_prompt(model_key: str, patient_message: str) -> str:
    if model_key == "llama3":
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"You are a Doctor conducting a diagnostic interview.\n"
            f"GOAL: Ask ONE brief question to help differentiate the patient's condition.\n"
            f"CRITICAL RULE: Output ONLY the question. No internal thoughts, no preamble.\n"
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Patient said: {patient_message}"
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    elif model_key == "gemma2":
        return (
            f"<start_of_turn>user\n"
            f"You are a Doctor conducting a diagnostic interview.\n"
            f"CURRENT INFO FROM PATIENT: {patient_message}\n"
            f"GOAL: Ask ONE brief question to help differentiate the patient's condition.\n"
            f"CRITICAL RULE: Output ONLY the question you want to ask.<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    elif model_key == "phi3":
        return (
            f"<|user|>\n"
            f"You are a Doctor conducting a diagnostic interview.\n"
            f"CURRENT INFO FROM PATIENT: {patient_message}\n"
            f"GOAL: Ask ONE brief question to help differentiate the patient's condition.\n"
            f"CRITICAL RULE: Output ONLY the question you want to ask.<|end|>\n"
            f"<|assistant|>\n"
        )
    else:  # biomistral / mistral
        return (
            f"[INST] You are a Doctor conducting a diagnostic interview.\n"
            f"CURRENT INFO FROM PATIENT: {patient_message}\n"
            f"GOAL: Ask ONE brief question to help differentiate the patient's condition.\n"
            f"CRITICAL RULE: Output ONLY the question you want to ask. [/INST]"
        )


# FIX 1: pathology_hint est désormais une simple chaîne dérivée du titre de la pathologie.
# On a supprimé l'appel à orchestrator.get_context(pathology, "hint") qui n'existe pas.
def build_diagnosis_prompt(model_key: str, full_history: str, pathology_hint: str) -> str:
    hint_line = f"HINT: {pathology_hint}\n" if pathology_hint else ""
    if model_key == "llama3":
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"You are a Doctor formulating a final diagnosis.\n"
            f"Review the interview history and state your FINAL DIAGNOSIS with a 2-sentence clinical justification.\n"
            f"{hint_line}"
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"HISTORY:\n{full_history}"
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    elif model_key == "gemma2":
        return (
            f"<start_of_turn>user\n"
            f"Review the following medical interview history.\n"
            f"State your FINAL DIAGNOSIS and provide a 2-sentence clinical justification.\n"
            f"{hint_line}\n"
            f"HISTORY:\n{full_history}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    elif model_key == "phi3":
        return (
            f"<|user|>\n"
            f"Review the following medical interview history.\n"
            f"State your FINAL DIAGNOSIS and provide a 2-sentence clinical justification.\n"
            f"{hint_line}\n"
            f"HISTORY:\n{full_history}<|end|>\n"
            f"<|assistant|>\n"
        )
    else:  # biomistral / mistral
        return (
            f"[INST] Review the following medical interview history.\n"
            f"State your FINAL DIAGNOSIS and provide a 2-sentence clinical justification.\n"
            f"{hint_line}\n"
            f"HISTORY:\n{full_history} [/INST]"
        )

# ==========================================
# 8. PROMPT BUILDERS — LLM JUDGE
# ==========================================
def build_judge_patient_content(patient_turns: list, patient_context: str) -> str:
    turns_text = "\n\n".join([
        f"[Turn {i+1}]\n{t['content']}"
        for i, t in enumerate(patient_turns)
    ])
    return f"""### CLINICAL CONTEXT PROVIDED TO THE PATIENT:
{patient_context}

### PATIENT RESPONSES TO EVALUATE:
{turns_text}

Please provide your evaluation now using the requested JSON format."""


def build_judge_doctor_content(doctor_turns: list, final_diagnosis: str, ground_truth: str) -> str:
    turns_text = "\n\n".join([
        f"[Question {i+1}]\n{t['content']}"
        for i, t in enumerate(doctor_turns)
    ])
    return f"""### CLINICAL TRUTH:
{ground_truth}

### DOCTOR QUESTIONS DURING INTERVIEW:
{turns_text}

### FINAL DIAGNOSIS GIVEN:
{final_diagnosis}

Please provide your evaluation now using the requested JSON format."""

# ==========================================
# 9. RESPONSE PARSER
# ==========================================
def parse_response(model_key: str, decoded_text: str) -> str:
    if model_key == "llama3":
        if "<|start_header_id|>assistant<|end_header_id|>" in decoded_text:
            response = decoded_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        else:
            response = decoded_text
        response = response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
    elif model_key == "gemma2":
        if "<start_of_turn>model" in decoded_text:
            response = decoded_text.split("<start_of_turn>model")[-1]
        else:
            response = decoded_text
        response = response.replace("<end_of_turn>", "").strip()
    elif model_key == "phi3":
        if "<|assistant|>" in decoded_text:
            response = decoded_text.split("<|assistant|>")[-1]
        else:
            response = decoded_text
        response = response.replace("<|end|>", "").strip()
    else:  # biomistral / mistral
        response = decoded_text.split("[/INST]")[-1].strip()

    for artifact in ["Patient:", "Jean:", "Answer:", "Doctor:", "Assistant:"]:
        response = response.replace(artifact, "")
    return response.strip()

# ==========================================
# 10. HEURISTIC SCORING
# ==========================================
def score_patient_turn(response: str, jargon_list: list = None) -> dict:
    active_jargon = jargon_list if jargon_list else MEDICAL_JARGON_FALLBACK

    text       = response.lower()
    word_count = len(response.split())

    if   50 <= word_count <= 200: length_score = 3
    elif 30 <= word_count <  50 or 200 < word_count <= 250: length_score = 2
    elif 10 <= word_count <  30: length_score = 1
    else: length_score = 0

    jargon_found = [j for j in active_jargon if j in text]
    jargon_score = max(0, 4 - len(jargon_found))

    role_leakage = any(phrase in text for phrase in DOCTOR_ROLE_LEAKAGE)
    role_score   = 0 if role_leakage else 3

    return {
        "length_score": length_score,
        "jargon_score": jargon_score,
        "role_score":   role_score,
        "jargon_found": jargon_found,
        "word_count":   word_count,
        "total":        length_score + jargon_score + role_score
    }


def score_doctor_turn(response: str, is_final_diagnosis: bool = False) -> dict:
    text       = response.lower()
    word_count = len(response.split())

    if is_final_diagnosis:
        disease_keywords = ["osteoarthritis", "arthritis", "rheumatoid", "gout", "joint", "diagnosis"]
        disease_score = 2 if any(k in text for k in disease_keywords) else 0

        if   30 <= word_count <= 200: length_score = 5
        elif word_count > 200:        length_score = 3
        else:                         length_score = 1

        role_leakage = any(phrase in text for phrase in PATIENT_ROLE_LEAKAGE)
        role_score   = 0 if role_leakage else 3

        return {
            "disease_score": disease_score,
            "length_score":  length_score,
            "role_score":    role_score,
            "word_count":    word_count,
            "total":         disease_score + length_score + role_score
        }
    else:
        question_score = 2 if "?" in response else 0

        if   word_count <= 80:  length_score = 3
        elif word_count <= 120: length_score = 2
        else:                   length_score = 1

        role_leakage   = any(phrase in text for phrase in PATIENT_ROLE_LEAKAGE)
        role_score     = 0 if role_leakage else 2

        clinical_keywords = [
            "pain", "joint", "morning", "stiffness", "swelling", "warm",
            "red", "movement", "family", "history", "better", "worse",
            "fever", "weight", "fatigue", "how long", "when", "where"
        ]
        clinical_score = 3 if any(k in text for k in clinical_keywords) else 0

        return {
            "question_score":  question_score,
            "length_score":    length_score,
            "role_score":      role_score,
            "clinical_score":  clinical_score,
            "word_count":      word_count,
            "total":           question_score + length_score + role_score + clinical_score
        }

# ==========================================
# 11. INFERENCE HELPERS
# ==========================================
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1800
    ).to("cuda")
    input_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(ids[0][input_length:], skip_special_tokens=True).strip()


def generate_judge(judge_model, judge_tok, system_prompt: str, user_content: str) -> str:
    combined    = f"{system_prompt}\n\n{user_content}"
    messages    = [{"role": "user", "content": combined}]
    prompt_text = judge_tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs       = judge_tok(prompt_text, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=judge_tok.eos_token_id
        )
    return judge_tok.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

# ==========================================
# 12. DYNAMIC JARGON GENERATION
# ==========================================
def generate_jargon_list(pathology: str, judge_model, judge_tok) -> list:
    print(f"\n🔤 Generating dynamic jargon list for '{pathology}'...")
    user_content = (
        f"Pathology: {pathology}\n\n"
        "List all medical/clinical terms that a real patient would NOT use in everyday speech "
        "when describing symptoms of this condition. Include terms specific to this pathology "
        "and general medical jargon a layperson would avoid."
    )
    raw    = generate_judge(judge_model, judge_tok, JARGON_SYSTEM_PROMPT, user_content)
    parsed = extract_json_from_text(raw)

    if parsed:
        terms = parsed.get("jargon_terms", [])
        if isinstance(terms, list) and len(terms) > 0:
            result = [t.lower().strip() for t in terms if isinstance(t, str)]
            print(f"   ✅ {len(result)} jargon terms generated: {result[:5]}...")
            return result

    print("   ⚠️  Jargon generation failed — using fallback list.")
    return MEDICAL_JARGON_FALLBACK

# ==========================================
# 13. LLM JUDGE — EVALUATE PATIENT & DOCTOR
# ==========================================
def run_llm_judge(result, patient_context, ground_truth, judge_model, judge_tok) -> dict:
    transcript    = result["transcript"]
    patient_turns = [m for m in transcript if m["role"] == "Patient"]
    doctor_turns  = [m for m in transcript if m["role"] == "Doctor"][1:]  # skip opening line
    final_diag    = next(
        (m["content"] for m in transcript if m["role"] == "Final Diagnosis"),
        "Not provided."
    )

    judge_results = {}

    print("   🧑‍⚕️ Judge evaluating PATIENT...")
    raw_patient    = generate_judge(judge_model, judge_tok, JUDGE_SYSTEM_PATIENT,
                                    build_judge_patient_content(patient_turns, patient_context))
    parsed_patient = extract_json_from_text(raw_patient)
    if parsed_patient:
        print(f"   ✅ Patient judge OK — score: {parsed_patient.get('total_score_out_of_10', '?')}/10")
        judge_results["llm_judge_patient"] = parsed_patient
    else:
        print("   ⚠️  Patient judge: JSON parsing failed.")
        judge_results["llm_judge_patient"] = {"raw_text_error": raw_patient}

    print("   🩺 Judge evaluating DOCTOR...")
    raw_doctor    = generate_judge(judge_model, judge_tok, JUDGE_SYSTEM_DOCTOR,
                                   build_judge_doctor_content(doctor_turns, final_diag, ground_truth))
    parsed_doctor = extract_json_from_text(raw_doctor)
    if parsed_doctor:
        print(f"   ✅ Doctor judge OK — score: {parsed_doctor.get('total_score_out_of_10', '?')}/10")
        judge_results["llm_judge_doctor"] = parsed_doctor
    else:
        print("   ⚠️  Doctor judge: JSON parsing failed.")
        judge_results["llm_judge_doctor"] = {"raw_text_error": raw_doctor}

    if parsed_patient and parsed_doctor:
        p = parsed_patient.get("total_score_out_of_10", 0) or 0
        d = parsed_doctor.get("total_score_out_of_10",  0) or 0
        judge_results["llm_judge_total_out_of_20"] = round(p + d, 2)
        print(f"   📊 LLM Judge Total: {judge_results['llm_judge_total_out_of_20']}/20")

    return judge_results

# ==========================================
# 14. SINGLE SIMULATION
# ==========================================
def run_simulation(patient_key, doctor_key,
                   patient_model, patient_tok,
                   doctor_model, doctor_tok,
                   orchestrator, pathology,
                   jargon_list=None):

    patient_context = orchestrator.get_context(pathology, "patient")

    # FIX 2: pathology_hint dérivé directement du titre de la pathologie.
    # L'Orchestrator ne supporte pas le rôle "hint" — on utilise le nom de la pathologie.
    pathology_hint = pathology

    transcript     = []
    patient_scores = []
    doctor_scores  = []

    print(f"\n{'='*60}")
    print(f"🚀 SIMULATION: Patient={patient_key.upper()} | Doctor={doctor_key.upper()} | Pathology={pathology}")
    print(f"{'='*60}")

    current_message = "Hello, I am the physician attending to you today. What symptoms are you experiencing?"
    transcript.append({"role": "Doctor", "content": current_message})
    print(f"🩺 [Doctor]: {current_message}")

    # --- PHASE 1: CLINICAL INTERVIEW ---
    for turn in range(TURNS):

        # PATIENT TURN
        patient_prompt   = build_patient_prompt(patient_key, patient_context, current_message)
        decoded          = generate(patient_model, patient_tok, patient_prompt,
                                    max_new_tokens=150, temperature=0.8)
        patient_response = parse_response(patient_key, decoded)
        patient_score    = score_patient_turn(patient_response, jargon_list)
        patient_scores.append(patient_score)

        print(f"👴 [Patient]: {patient_response}")
        print(f"   → Heuristic: {patient_score['total']}/10 | "
              f"words={patient_score['word_count']} | jargon={patient_score['jargon_found']}")
        transcript.append({"role": "Patient", "content": patient_response, "score": patient_score})
        current_message = patient_response

        if turn == TURNS - 1:
            break

        # DOCTOR TURN
        doc_prompt      = build_doctor_prompt(doctor_key, current_message)
        decoded         = generate(doctor_model, doctor_tok, doc_prompt,
                                   max_new_tokens=120, temperature=0.3)
        doctor_response = parse_response(doctor_key, decoded)
        doctor_score    = score_doctor_turn(doctor_response, is_final_diagnosis=False)
        doctor_scores.append(doctor_score)

        print(f"🩺 [Doctor]: {doctor_response}")
        print(f"   → Heuristic: {doctor_score['total']}/10")
        transcript.append({"role": "Doctor", "content": doctor_response, "score": doctor_score})
        current_message = doctor_response

    # --- PHASE 2: FINAL DIAGNOSIS ---
    print("\n🧐 [Doctor formulating final diagnosis...]")
    history_lines = [f"{m['role']}: {m['content']}" for m in transcript]
    full_history  = "\n".join(history_lines)
    words = full_history.split()
    if len(words) > 600:
        full_history = " ".join(words[-600:])

    final_prompt    = build_diagnosis_prompt(doctor_key, full_history, pathology_hint)
    decoded         = generate(doctor_model, doctor_tok, final_prompt,
                               max_new_tokens=300, temperature=0.1)
    final_diagnosis = parse_response(doctor_key, decoded)
    final_score     = score_doctor_turn(final_diagnosis, is_final_diagnosis=True)
    doctor_scores.append(final_score)

    print(f"\n🩺 [FINAL DIAGNOSIS]: {final_diagnosis}")
    print(f"   → Heuristic: {final_score['total']}/10")
    transcript.append({"role": "Final Diagnosis", "content": final_diagnosis, "score": final_score})

    avg_patient = sum(s["total"] for s in patient_scores) / len(patient_scores) if patient_scores else 0
    avg_doctor  = sum(s["total"] for s in doctor_scores)  / len(doctor_scores)  if doctor_scores  else 0
    total_20    = round(avg_patient + avg_doctor, 2)

    print(f"\n📊 HEURISTIC — Patient avg: {avg_patient:.1f}/10 | "
          f"Doctor avg: {avg_doctor:.1f}/10 | Total: {total_20}/20")

    return {
        "pathology_target":          pathology,
        "patient_model":             MODELS[patient_key]["hf_path"],
        "doctor_model":              MODELS[doctor_key]["hf_path"],
        "patient_model_key":         patient_key,
        "doctor_model_key":          doctor_key,
        "transcript":                transcript,
        "scores": {
            "patient_avg_score":     round(avg_patient, 2),
            "doctor_avg_score":      round(avg_doctor,  2),
            "total_grade_out_of_20": total_20,
            "patient_turn_scores":   patient_scores,
            "doctor_turn_scores":    doctor_scores,
        },
        "llm_judge_patient":         None,
        "llm_judge_doctor":          None,
        "llm_judge_total_out_of_20": None,
    }

# ==========================================
# 15. BENCHMARK RUNNER (per patient model)
# ==========================================
def run_benchmark_for_patient(patient_key, doctor_list,
                               patient_model, patient_tok,
                               judge_model, judge_tok,
                               orchestrator, pathologies,
                               jargon_lists, use_judge):
    for doctor_key, config_idx in doctor_list:

        if doctor_key == patient_key:
            doctor_model, doctor_tok = patient_model, patient_tok
            same_model = True
        elif use_judge and doctor_key == JUDGE_MODEL_KEY:
            doctor_model, doctor_tok = judge_model, judge_tok
            same_model = True
        else:
            doctor_model, doctor_tok = load_model(doctor_key)
            same_model = False

        for pathology in pathologies:
            output_dir = os.path.join(
                MODELS[patient_key]["output_dir"],
                f"patient_{patient_key}_doctor_{doctor_key}",
                pathology.replace(" ", "_").lower()
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "transcript.json")

            if os.path.exists(output_path):
                print(f"\n⏭️  [{config_idx:2d}] {patient_key.upper()} × {doctor_key.upper()} "
                      f"| {pathology} — skipping.")
                continue

            result = run_simulation(
                patient_key, doctor_key,
                patient_model, patient_tok,
                doctor_model, doctor_tok,
                orchestrator, pathology,
                jargon_list=jargon_lists.get(pathology)
            )

            if use_judge:
                print(f"\n🧑‍⚖️  Running LLM Judge ({JUDGE_MODEL_KEY.upper()})...")
                patient_context = orchestrator.get_context(pathology, "patient")
                ground_truth    = orchestrator.get_context(pathology, "judge")
                result.update(run_llm_judge(
                    result, patient_context, ground_truth,
                    judge_model, judge_tok
                ))

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"✅ Saved → {output_path}")

        if not same_model:
            print(f"🧹 Unloading doctor {doctor_key.upper()}...")
            unload_model(doctor_model, doctor_tok)

# ==========================================
# 16. MAIN
# ==========================================
def main():
    args = parse_args()

    if args.list:
        print("\nAvailable configurations:")
        for i, (p, d) in enumerate(ALL_CONFIGS):
            print(f"  {i+1:2d}. patient={p.upper():12s} | doctor={d.upper()}")
        return

    selected_indices = (list(range(len(ALL_CONFIGS))) if args.configs == "all"
                        else parse_config_selection(args.configs))
    selected_configs = [(ALL_CONFIGS[i], i + 1)
                        for i in selected_indices if i < len(ALL_CONFIGS)]

    print(f"\n🏭 BENCHMARK — {len(selected_configs)} configuration(s) selected")
    for (p, d), idx in selected_configs:
        print(f"  [{idx:2d}] patient={p.upper()} | doctor={d.upper()}")

    use_judge = not args.no_judge
    print(f"\n{'🧑‍⚖️  LLM Judge enabled — ' + JUDGE_MODEL_KEY.upper() if use_judge else '⚡ LLM Judge disabled (--no-judge)'}")

    # FIX 3: Suppression du bloc orphelin mal indenté (boucle for pathology hors de toute fonction).
    # La découverte des pathologies reste correcte ici.
    orchestrator = MedSimOrchestrator(file_path=DATA_PATH)
    pathologies  = list({item["title"] for item in orchestrator.db})
    print(f"\n📚 Pathologies found in knowledge base: {pathologies}")

    judge_model, judge_tok = None, None
    if use_judge:
        print(f"\n{'#'*70}\n# Pre-loading JUDGE: {JUDGE_MODEL_KEY.upper()}\n{'#'*70}")
        judge_model, judge_tok = load_model(JUDGE_MODEL_KEY)

    jargon_lists = {}
    for pathology in pathologies:
        if use_judge:
            jargon_lists[pathology] = generate_jargon_list(pathology, judge_model, judge_tok)
        else:
            print(f"   ℹ️  Using fallback jargon list for '{pathology}' (--no-judge mode).")
            jargon_lists[pathology] = MEDICAL_JARGON_FALLBACK

    by_patient = defaultdict(list)
    for (p, d), idx in selected_configs:
        by_patient[p].append((d, idx))

    for patient_key, doctor_list in by_patient.items():
        print(f"\n{'#'*70}\n# Loading PATIENT: {patient_key.upper()}\n{'#'*70}")

        if use_judge and patient_key == JUDGE_MODEL_KEY:
            patient_model, patient_tok = judge_model, judge_tok
            own_patient_model = False
        else:
            patient_model, patient_tok = load_model(patient_key)
            own_patient_model = True

        run_benchmark_for_patient(
            patient_key, doctor_list,
            patient_model, patient_tok,
            judge_model, judge_tok,
            orchestrator, pathologies,
            jargon_lists, use_judge
        )

        if own_patient_model:
            print(f"\n🧹 Unloading patient {patient_key.upper()}...")
            unload_model(patient_model, patient_tok)

    if use_judge and judge_model is not None:
        print(f"\n🧹 Unloading judge {JUDGE_MODEL_KEY.upper()}...")
        unload_model(judge_model, judge_tok)

    print("\n🎉 BENCHMARK COMPLETED!")


if __name__ == "__main__":
    main()