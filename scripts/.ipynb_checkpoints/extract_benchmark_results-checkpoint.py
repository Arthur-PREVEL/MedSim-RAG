import os
import json
import csv
import glob

# ==========================================
# PATH CONFIGURATION
# ==========================================
RESULTS_DIR = "../results/benchmark/"
OUTPUT_CSV = "../results/ultimate_benchmark_summary.csv"

# List of models evaluated in the pipeline
MODELS = ["llama3", "mistral", "biomistral", "gemma2", "phi3"]

def extract_results():
    print("📊 STARTING MULTI-MODEL RESULTS EXTRACTION...")
    
    # Use the first model's directory as a reference for the case files list
    reference_dir = os.path.join(RESULTS_DIR, MODELS[0])
    files = glob.glob(os.path.join(reference_dir, "transcript_*.json"))
    if not files:
        print(f"❌ No files found in reference directory: {reference_dir}")
        return

    csv_data = []

    for filepath in files:
        filename = os.path.basename(filepath)
        
        # Load base reference data to fetch the clinical ground truth and transcript details
        with open(filepath, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
            
        pathology = base_data.get("pathology_target", "Unknown")
        final_diagnosis = next((m['content'] for m in base_data.get('transcript', []) if m['role'] == 'Final Diagnosis'), "N/A")

        # Initialize the row dictionary for the CSV/Excel spreadsheet
        row = {
            "Pathology (Ground Truth)": pathology,
            "AI Final Diagnosis": final_diagnosis,
        }

        # Dynamically loop through each model directory to gather scores and justifications
        for model in MODELS:
            model_filepath = os.path.join(RESULTS_DIR, model, filename)
            model_bench = {}
            
            if os.path.exists(model_filepath):
                with open(model_filepath, 'r', encoding='utf-8') as mf:
                    full_data = json.load(mf)
                    model_bench = full_data.get(f"benchmark_{model}", {})

            # If JSON parsing failed during the benchmark run, fallback safely
            if "raw_text_error" in model_bench:
                row[f"{model.upper()} - Error"] = "JSON PARSING FAILED"
                row[f"{model.upper()} - Total Grade (/20)"] = "N/A"
            else:
                row[f"{model.upper()} - Accuracy (/5)"] = model_bench.get("diagnosis_accuracy", {}).get("score_out_of_5", "")
                row[f"{model.upper()} - Reasoning (/10)"] = model_bench.get("clinical_reasoning", {}).get("score_out_of_10", "")
                row[f"{model.upper()} - Safety (/5)"] = model_bench.get("patient_safety", {}).get("score_out_of_5", "")
                row[f"{model.upper()} - Total Grade (/20)"] = model_bench.get("total_grade_out_of_20", "")
                row[f"{model.upper()} - Feedback"] = model_bench.get("final_feedback", "")

        csv_data.append(row)

    # Save data into a structured CSV file
    headers = csv_data[0].keys()
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter=';')
        writer.writeheader()
        writer.writerows(csv_data)

    print("✅ Extraction completed successfully!")
    print(f"💾 File generated: {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_results()