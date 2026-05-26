"""
compile_results.py
------------------
Reads all evaluation JSON files from the results directory
and compiles them into a single CSV for analysis.

Usage: python compile_results.py
Output: ../results/comparison_results.csv
"""

import os
import sys
import json
import csv
import re

RESULTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
OUTPUT_CSV = os.path.join(RESULTS_ROOT, "comparison_results.csv")

def extract_scores(evaluation_text):
    """Extract individual criterion scores from evaluation text."""
    scores = {}

    # Try "Score: X/5" format first (Llama judge)
    score_matches = re.findall(r'(\d+)\.\s*(.*?):.*?Score:\s*(\d+)/5', evaluation_text, re.DOTALL)
    if score_matches:
        for _, criterion, score in score_matches:
            key = criterion.strip()[:40]
            scores[key] = int(score)
    else:
        # Fallback: extract X/5 patterns in order
        raw_scores = re.findall(r'(\d+)/5', evaluation_text)
        labels = ["Diagnosis", "Key Symptoms", "Interview Technique", "Clinical Justification"]
        for i, label in enumerate(labels):
            if i < len(raw_scores):
                scores[label] = int(raw_scores[i])

    # Extract total grade
    total_match = re.search(r'Total Grade[:\s]+(\d+)/20', evaluation_text)
    total = int(total_match.group(1)) if total_match else None

    # Recalculate total if missing
    if total is None and len(scores) >= 4:
        total = sum(list(scores.values())[:4])

    return scores, total

def find_evaluation_files(results_root):
    """Walk results directory and find all evaluation JSON files."""
    found = []
    for dirpath, _, filenames in os.walk(results_root):
        for fname in filenames:
            if fname.startswith("evaluation_judge_") and fname.endswith(".json"):
                found.append(os.path.join(dirpath, fname))
    return sorted(found)

def main():
    eval_files = find_evaluation_files(RESULTS_ROOT)

    if not eval_files:
        print(f"❌ No evaluation files found in {RESULTS_ROOT}")
        print("   Run run_all_experiments.py first.")
        return

    print(f"📂 Found {len(eval_files)} evaluation file(s):")
    for f in eval_files:
        print(f"   {f}")

    rows = []
    for eval_path in eval_files:
        with open(eval_path, "r") as f:
            data = json.load(f)

        scores, total = extract_scores(data.get("evaluation_text", ""))

        # Load transcript to count patient turns and avg length
        transcript_path = os.path.join(os.path.dirname(eval_path), "session_transcript.json")
        avg_patient_length = None
        if os.path.exists(transcript_path):
            with open(transcript_path, "r") as f:
                transcript_data = json.load(f)
            patient_turns = [
                m['content'] for m in transcript_data['transcript']
                if m['role'] == 'Patient'
            ]
            if patient_turns:
                avg_patient_length = round(
                    sum(len(t.split()) for t in patient_turns) / len(patient_turns), 1
                )

        row = {
            "Pathology": data.get("pathology_target", ""),
            "Patient Model": data.get("patient_model", "").split("/")[-1],
            "Doctor Model": data.get("doctor_model", "").split("/")[-1],
            "Judge Model": data.get("judge_model", "").split("/")[-1],
            "Diagnosis Score (/5)": scores.get("Diagnosis", scores.get(list(scores.keys())[0], "") if scores else ""),
            "Key Symptoms Score (/5)": scores.get("Key Symptoms", scores.get(list(scores.keys())[1], "") if len(scores) > 1 else ""),
            "Interview Technique (/5)": scores.get("Interview Technique", scores.get(list(scores.keys())[2], "") if len(scores) > 2 else ""),
            "Clinical Justification (/5)": scores.get("Clinical Justification", scores.get(list(scores.keys())[3], "") if len(scores) > 3 else ""),
            "Total Grade (/20)": total if total is not None else data.get("total_grade", ""),
            "Avg Patient Response (words)": avg_patient_length,
        }
        rows.append(row)

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Results compiled → {OUTPUT_CSV}")
    print(f"\n📊 Summary:")
    print(f"{'Patient':<25} {'Doctor':<25} {'Judge':<25} {'Total /20'}")
    print("-" * 90)
    for row in rows:
        print(f"{row['Patient Model']:<25} {row['Doctor Model']:<25} {row['Judge Model']:<25} {row['Total Grade (/20)']}")

if __name__ == "__main__":
    main()