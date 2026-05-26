"""
run_all_experiments.py
----------------------
Runs all simulation + judge combinations sequentially.
Each experiment is:  (patient_model, doctor_model, judge_model)

Configurations:
  A: BioMistral patient | BioMistral doctor | BioMistral judge  (baseline)
  B: Llama patient      | BioMistral doctor | BioMistral judge  (isolate patient)
  C: Llama patient      | Llama doctor      | BioMistral judge  (isolate doctor)
  D: Llama patient      | BioMistral doctor | Llama judge       (isolate judge)
  E: Llama patient      | Llama doctor      | Llama judge       (full Llama)

Run all:        python run_all_experiments.py
Run one config: python run_all_experiments.py --configs A B
"""

import subprocess
import sys
import argparse
import os

EXPERIMENTS = {
    "A": {
        "label": "Full BioMistral (baseline)",
        "patient": "biomistral",
        "doctor":  "biomistral",
        "judge":   "biomistral",
        "result_dir": "../results/patient_biomistral_doctor_biomistral",
    },
    "B": {
        "label": "Llama patient | BioMistral doctor | BioMistral judge",
        "patient": "llama",
        "doctor":  "biomistral",
        "judge":   "biomistral",
        "result_dir": "../results/patient_llama_doctor_biomistral",
    },
    "C": {
        "label": "Llama patient | Llama doctor | BioMistral judge",
        "patient": "llama",
        "doctor":  "llama",
        "judge":   "biomistral",
        "result_dir": "../results/patient_llama_doctor_llama",
    },
    "D": {
        "label": "Llama patient | BioMistral doctor | Llama judge",
        "patient": "llama",
        "doctor":  "biomistral",
        "judge":   "llama",
        "result_dir": "../results/patient_llama_doctor_biomistral",
    },
    "E": {
        "label": "Full Llama",
        "patient": "llama",
        "doctor":  "llama",
        "judge":   "llama",
        "result_dir": "../results/patient_llama_doctor_llama",
    },
}

def run_command(cmd):
    print(f"\n▶ Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode

def parse_args():
    parser = argparse.ArgumentParser(description="Run all MedSim experiments")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=EXPERIMENTS.keys(),
        default=list(EXPERIMENTS.keys()),
        help="Which configs to run (default: all)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    print("="*60)
    print("🧪 MEDSIM-RAG EXPERIMENT RUNNER")
    print(f"   Running configs: {', '.join(args.configs)}")
    print("="*60)

    for config_id in args.configs:
        exp = EXPERIMENTS[config_id]
        print(f"\n{'='*60}")
        print(f"📋 Config {config_id}: {exp['label']}")
        print(f"{'='*60}")

        # Step 1: Run simulation
        sim_cmd = [
            sys.executable,
            os.path.join(scripts_dir, "run_sim.py"),
            "--patient", exp["patient"],
            "--doctor",  exp["doctor"],
        ]
        try:
            run_command(sim_cmd)
        except subprocess.CalledProcessError as e:
            print(f"❌ Simulation failed for config {config_id}: {e}")
            continue

        # Step 2: Run judge
        transcript_path = os.path.join(exp["result_dir"], "session_transcript.json")
        judge_cmd = [
            sys.executable,
            os.path.join(scripts_dir, "run_judge.py"),
            "--judge",      exp["judge"],
            "--transcript", transcript_path,
        ]
        try:
            run_command(judge_cmd)
        except subprocess.CalledProcessError as e:
            print(f"❌ Judge failed for config {config_id}: {e}")
            continue

        print(f"\n✅ Config {config_id} complete.")

    print("\n" + "="*60)
    print("🎉 All experiments done. Run compile_results.py to generate the CSV.")
    print("="*60)

if __name__ == "__main__":
    main()