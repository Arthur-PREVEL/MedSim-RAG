import os
import json
import csv

# ==========================================
# PATH CONFIGURATION
# ==========================================
RESULTS_DIR = "../results/benchmark_simulation/"
OUTPUT_CSV  = "../results/simulation_benchmark_summary.csv"

MODELS = ["biomistral", "llama3", "mistral", "gemma2", "phi3"]

# ==========================================
# HELPERS
# ==========================================
def safe_get(d: dict, *keys, default=""):
    """Safely navigate a nested dict without raising."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d != {} else default


def count_jargon(transcript: list) -> float:
    patient_turns = [m for m in transcript if m.get("role") == "Patient"]
    if not patient_turns:
        return 0.0
    total_jargon = sum(
        len(m.get("score", {}).get("jargon_found", []))
        for m in patient_turns
    )
    return round(total_jargon / len(patient_turns), 2)


def count_questions(transcript: list) -> int:
    doctor_turns = [m for m in transcript if m.get("role") == "Doctor"]
    return sum(1 for m in doctor_turns if "?" in m.get("content", ""))


def avg_word_count(transcript: list, role: str) -> float:
    turns = [m for m in transcript if m.get("role") == role]
    if not turns:
        return 0.0
    return round(
        sum(len(m.get("content", "").split()) for m in turns) / len(turns),
        1
    )


def turns_completed(transcript: list) -> int:
    return sum(1 for m in transcript if m.get("role") == "Patient")


def extract_llm_judge(data: dict, role: str) -> dict:
    key   = f"llm_judge_{role}"
    block = data.get(key)

    if block is None or "raw_text_error" in (block or {}):
        if role == "patient":
            return {
                "LLM Patient - Natural Language (/4)": "",
                "LLM Patient - Symptom Coherence (/3)": "",
                "LLM Patient - Role Adherence (/3)": "",
                "LLM Patient - Total (/10)": "",
                "LLM Patient - Feedback": "",
            }
        else:
            return {
                "LLM Doctor - Question Quality (/4)": "",
                "LLM Doctor - Diagnostic Accuracy (/3)": "",
                "LLM Doctor - Role Adherence (/3)": "",
                "LLM Doctor - Total (/10)": "",
                "LLM Doctor - Feedback": "",
            }

    if role == "patient":
        return {
            "LLM Patient - Natural Language (/4)": safe_get(block, "natural_language", "score_out_of_4"),
            "LLM Patient - Symptom Coherence (/3)": safe_get(block, "symptom_coherence", "score_out_of_3"),
            "LLM Patient - Role Adherence (/3)": safe_get(block, "role_adherence", "score_out_of_3"),
            "LLM Patient - Total (/10)": block.get("total_score_out_of_10", ""),
            "LLM Patient - Feedback": block.get("overall_feedback", ""),
        }
    else:
        return {
            "LLM Doctor - Question Quality (/4)": safe_get(block, "question_quality", "score_out_of_4"),
            "LLM Doctor - Diagnostic Accuracy (/3)": safe_get(block, "diagnostic_accuracy", "score_out_of_3"),
            "LLM Doctor - Role Adherence (/3)": safe_get(block, "role_adherence", "score_out_of_3"),
            "LLM Doctor - Total (/10)": block.get("total_score_out_of_10", ""),
            "LLM Doctor - Feedback": block.get("overall_feedback", ""),
        }

# ==========================================
# MAIN EXTRACTION (FIXED)
# ==========================================
def extract_results():
    print("📊 STARTING SIMULATION BENCHMARK EXTRACTION...")
    csv_data = []

    for patient_key in MODELS:
        for doctor_key in MODELS:

            # ✅ FIX IMPORTANT : structure réelle = patient/doctor/pathology/
            base_dir = os.path.join(
                RESULTS_DIR,
                patient_key,
                f"patient_{patient_key}_doctor_{doctor_key}"
            )

            if not os.path.exists(base_dir):
                print(f"⚠️ Missing folder: {base_dir}")
                continue

            # 🔍 scan toutes les pathologies
            found_any = False

            for root, _, files in os.walk(base_dir):
                if "transcript.json" not in files:
                    continue

                filepath = os.path.join(root, "transcript.json")

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"⚠️ Error reading {filepath}: {e}")
                    continue

                transcript = data.get("transcript", [])
                scores     = data.get("scores", {})

                row = {
                    "Config": f"{patient_key.upper()} × {doctor_key.upper()}",
                    "Patient Model": patient_key.upper(),
                    "Doctor Model": doctor_key.upper(),
                    "Pathology": data.get("pathology_target", "Unknown"),

                    # Heuristics
                    "Heuristic Patient (/10)": safe_get(scores, "patient_avg_score", default=""),
                    "Heuristic Doctor (/10)": safe_get(scores, "doctor_avg_score", default=""),
                    "Heuristic Total (/20)": safe_get(scores, "total_grade_out_of_20", default=""),

                    # LLM judge
                    "LLM Judge Total (/20)": data.get("llm_judge_total_out_of_20", ""),

                    # Metrics
                    "Patient Avg Words": avg_word_count(transcript, "Patient"),
                    "Patient Jargon/Turn": count_jargon(transcript),
                    "Patient Turns": turns_completed(transcript),
                    "Doctor Avg Words": avg_word_count(transcript, "Doctor"),
                    "Doctor Questions": count_questions(transcript),

                    # Final diagnosis
                    "Final Diagnosis": next(
                        (m["content"] for m in transcript if m["role"] == "Final Diagnosis"),
                        "N/A"
                    ),
                }

                row.update(extract_llm_judge(data, "patient"))
                row.update(extract_llm_judge(data, "doctor"))

                csv_data.append(row)
                found_any = True

                print(
                    f"✅ loaded: {patient_key} × {doctor_key} | "
                    f"{os.path.basename(root)}"
                )

            if not found_any:
                print(f"⚠️ No transcripts found for {patient_key} × {doctor_key}")

    if not csv_data:
        print("❌ No data found. Check RESULTS_DIR structure.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\n✅ Extraction complete!")
    print(f"💾 CSV saved → {OUTPUT_CSV}")
    print(f"📋 {len(csv_data)} configurations extracted.")


if __name__ == "__main__":
    extract_results()