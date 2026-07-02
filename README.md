![License: Proprietary](https://img.shields.io/badge/License-Proprietary_&_All_Rights_Reserved-red.svg)

> **Copyright Notice:** This repository is a portfolio showcase. The code is made available for viewing purposes only. No reproduction, modification, or distribution is allowed without explicit written permission from the authors. See the `LICENSE` file for more details.

<div align="center">
  <h1>
    <img src="images/medsimlogo.png" alt="MedSim Logo" width="40" style="vertical-align: middle; margin-right: 10px;"/>
    MedSim: High-Fidelity Clinical Simulator
  </h1>
  <p><i>An AI-powered medical simulation ecosystem secured by RAG and evaluated via an MLOps benchmarking pipeline.</i></p>
</div>

---

## Project Demonstration

The following video demonstrates the complete system architecture in action, highlighting the dynamic VRAM swapping mechanism and the customizable Multi-Agent Engine. The interface allows seamless transition between various Instruction-Tuned models (Llama-3, Gemma-2, Mistral, BioMistral, Phi-3) for both the Patient Agent and the Supervisor Agent.

<div align="center">
  <a href="https://youtu.be/sl_wyVNGyYw?si=X8N37PXTsvnKNsEK">
    <img src="https://img.youtube.com/vi/sl_wyVNGyYw/maxresdefault.jpg" alt="MedSim Demo Video" width="700" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
  </a>
  <br>
  <p><i>Click the image above to watch the MedSim Demo on YouTube</i></p>
</div>

---

## Executive Summary

Clinical interviewing is a cornerstone of medical practice, yet scalable environments for students to practice diagnostic formulation and receive immediate expert feedback remain limited. **MedSim** addresses this gap by providing a dual-agent simulation environment:

1. **The Patient Agent:** A Retrieval-Augmented Generation (RAG) LLM that simulates realistic symptoms based on validated medical literature. It is strictly prompted to avoid medical jargon and prevent diagnostic leakage.
2. **The Supervisor Agent (LLM-as-a-Judge):** A highly analytical evaluator that grades the student's clinical reasoning, diagnostic accuracy, and patient safety protocols, outputting deterministic JSON reports.

## Technical Architecture & Innovations

### Dynamic VRAM Swapping
A core constraint of the project was running multiple heavily quantized 7B-9B parameter models concurrently on a single NVIDIA Tesla V100 (16GB GPU) to optimize infrastructure costs. To prevent Out-Of-Memory (OOM) crashes, we engineered a state-managed Dynamic VRAM Swapping protocol. The memory is explicitly purged (`torch.cuda.empty_cache()`) and models are swapped on the fly between the interview phase and the grading phase.

### MLOps Evaluation Factory (The Tri-Agent Sandbox)
To validate the pedagogical accuracy of the system at scale, we developed an automated benchmarking pipeline. By replacing the human user with a "Doctor Agent", we ran a combinatorial evaluation matrix across 5 open-weight models.

This automated metrology phase allowed us to:
* Evaluate model reasoning capabilities and context window limits across complex clinical scenarios.
* Quantitatively measure and mitigate LLM vulnerabilities, such as compliance bias (sycophancy) and medical hallucinations.
* Identify the most rigorous LLM pairings for production deployment.

## Technical Stack

* **Language:** Python 3.11
* **Deep Learning Frameworks:** PyTorch (CUDA 12.1), Hugging Face Transformers, Accelerate
* **Optimization:** BitsAndBytes (4-bit NF4 Quantization)
* **Frontend:** Streamlit
* **Data Engineering:** BeautifulSoup (ETL parsing), Pandas

---

## Installation & Setup

### 1. Environment Preparation

To ensure compatibility with CUDA 12.1 and specific hardware optimizations, deploying within an isolated Conda environment is highly recommended.

```bash
conda create -n medsim_env python=3.11
conda activate medsim_env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage Guide

The repository is structured around three main execution phases.

### Phase 1: Data Engineering (ETL)

To respect repository size limits and data licensing, the full database of 4,500+ pathologies is not distributed. You must run the scraper to build the complete RAG database locally, using the provided `pubmed-statpearls-set.txt` seed.

```bash
python etl/web_scrapping.py
```

*(Note: This pipeline enforces rate limits to respect NCBI server policies and will take several hours. A lightweight `knowledge_base_extract.json` is provided for immediate UI testing).*

### Phase 2: Interactive Web Application

To launch the client-facing clinical simulator:

```bash
streamlit run app.py
```

### Phase 3: MLOps Benchmarking Pipeline

To run the automated model evaluation factory and generate the statistical metrics:

1. **Execute the Multi-Agent Interactions:**

```bash
python scripts/run_benchmark.py
```

2. **Data Aggregation:** Parse the generated JSON outputs into a centralized CSV dataset.

```bash
python scripts/extract_results.py
```

3. **Data Visualization:** Generate the performance heatmaps and variance boxplots (outputs saved to `/results/graphs/`).

```bash
python scripts/generate_graphs.py
```

---

## Authors & Context

* **Arthur Prevel** (ISIS Castres)
* **Aloïs Kamber** (CESI)

Developed as part of the **INF 3600 - Generative Artificial Intelligence** course at **UiT The Arctic University of Norway** (Tromsø). Data sourced ethically from the NCBI StatPearls Database.
