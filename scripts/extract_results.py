import os
import glob
import json
import csv

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_DIR = "../results/biomistral/"
OUTPUT_CSV = "../results/biomistral_summary.csv"

def extract_data():
    print(f"🔍 Searching for JSON files in {RESULTS_DIR}...")
    json_files = glob.glob(os.path.join(RESULTS_DIR, "transcript_*.json"))
    
    if not json_files:
        print("❌ No transcript files found. Check the path.")
        return

    print(f"📄 {len(json_files)} files found. Starting extraction...")

    # Preparing data for the CSV
    extracted_data = []
    
    # Defining our table columns
    headers = [
        "Pathology (Ground Truth)", 
        "Patient Model", 
        "Doctor Model", 
        "AI Final Diagnosis", 
        "Judge Evaluation (Raw Text)"
    ]

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 1. Extracting ground truth and models
        pathology = data.get("pathology_target", "Unknown")
        patient_model = data.get("patient_model", "Unknown")
        doctor_model = data.get("doctor_model", "Unknown")
        
        # 2. Extracting the diagnosis formulated by the Doctor
        final_diag = "N/A"
        for message in data.get("transcript", []):
            if message.get("role") == "Final Diagnosis":
                final_diag = message.get("content", "").replace('\n', ' ') # Removing line breaks for CSV compatibility
                break
                
        # 3. Extracting the Judge's evaluation
        evaluation = data.get("judge_evaluation", "Not Evaluated")
        evaluation_clean = evaluation.replace('\n', ' | ') # Replacing line breaks with a visual separator
        
        # Adding the row to the table
        extracted_data.append([
            pathology, 
            patient_model, 
            doctor_model, 
            final_diag, 
            evaluation_clean
        ])

    # Saving to CSV file
    print(f"\n💾 Saving data to {OUTPUT_CSV}...")
    
    # Using utf-8-sig so Excel reads special characters correctly (BOM)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";") # The semicolon prevents column bugs in European Excel
        writer.writerow(headers)
        writer.writerows(extracted_data)

    print("✨ Extraction completed successfully!")

if __name__ == "__main__":
    extract_data()