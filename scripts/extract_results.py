import os
import glob
import json
import csv

# ==========================================
# CONFIGURATION
# ==========================================
# Pointing to the new folder where the fully graded files are saved
RESULTS_DIR = "../results/llama3/"
OUTPUT_CSV = "../results/comparative_summary.csv"

def extract_data():
    print(f"🔍 Searching for JSON files in {RESULTS_DIR}...")
    json_files = glob.glob(os.path.join(RESULTS_DIR, "transcript_*.json"))
    
    if not json_files:
        print("❌ No transcript files found. Check the path.")
        return

    print(f"📄 {len(json_files)} files found. Starting extraction...")

    extracted_data = []
    
    # 🌟 HEADERS: Side-by-side comparison
    headers = [
        "Pathology (Ground Truth)", 
        "Patient Model", 
        "Doctor Model", 
        "AI Final Diagnosis", 
        "BioMistral Evaluation",
        "Llama-3 Evaluation"
    ]

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        pathology = data.get("pathology_target", "Unknown")
        patient_model = data.get("patient_model", "Unknown")
        doctor_model = data.get("doctor_model", "Unknown")
        
        final_diag = "N/A"
        for message in data.get("transcript", []):
            if message.get("role") == "Final Diagnosis":
                final_diag = message.get("content", "").replace('\n', ' ')
                break
                
        # Extract BioMistral evaluation
        bm_eval = data.get("judge_evaluation", "Not Evaluated")
        bm_eval_clean = bm_eval.replace('\n', ' | ')
        
        # Extract Llama-3 evaluation
        llama_eval = data.get("llama3_evaluation", "Not Evaluated")
        llama_eval_clean = llama_eval.replace('\n', ' | ')
        
        extracted_data.append([
            pathology, 
            patient_model, 
            doctor_model, 
            final_diag, 
            bm_eval_clean,
            llama_eval_clean
        ])

    print(f"\n💾 Saving comparative data to {OUTPUT_CSV}...")
    
    # Using utf-8-sig so Excel reads special characters correctly
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(headers)
        writer.writerows(extracted_data)

    print("✨ Extraction completed successfully! Open comparative_summary.csv to see the results.")

if __name__ == "__main__":
    extract_data()