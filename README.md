<div align="center">
  <h1>
    <img src="./images/medsim_logo.png" alt="MedSim Logo" width="40" style="vertical-align: middle; margin-right: 10px;"/>
    MedSim: High-Fidelity Clinical Simulator
  </h1>
  <p><i>An AI-powered medical simulation ecosystem secured by RAG and evaluated via MLOps benchmarking.</i></p>
</div>

---

## 🎥 Project Demonstration

Before diving into the code, watch our full system demonstration showcasing the dynamic VRAM swapping and our customizable Multi-Agent Engine, allowing you to seamlessly switch between 5 different LLMs (Llama-3, Gemma-2, Mistral, BioMistral, and Phi-3) for both the RAG-augmented Patient and the Virtual Professor:

<div align="center">
  <a href="https://youtu.be/M3fey_6fyeY">
    <img src="https://img.youtube.com/vi/M3fey_6fyeY/maxresdefault.jpg" alt="MedSim Demo Video" width="800" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"/>
  </a>
  <br>
  <p><i>👉 Click the image above to watch the MedSim Demo on YouTube 👈</i></p>
</div>

---

## 📖 Overview

Clinical interviewing is a cornerstone of medical practice, yet students lack a "safe space" to make clinical errors and receive immediate expert feedback. **MedSim** solves this by providing a dual-agent simulation environment:

1. **The Patient Agent:** A RAG-augmented LLM that simulates realistic symptoms based on validated medical literature, stripped of medical jargon.
2. **The Supervisor Agent (LLM-as-a-Judge):** A highly analytical evaluator that grades the student's clinical reasoning, diagnostic accuracy, and patient safety protocols.

This project was developed as part of the **INF 3600 - Generative Artificial Intelligence** course at **UiT The Arctic University of Norway**.

---

## 🔬 Research & Metrology: The Tri-Agent Sandbox

A massive portion of the workload for this project was dedicated to rigorous MLOps benchmarking. Validating a medical AI system manually is methodologically flawed and impossible to scale. To solve this, we engineered a fully automated **Tri-Agent Sandbox**.

Before deploying the dual-agent application for human users, we replaced the human medical student with an automated **"Doctor Agent"**. By deploying a combinatorial evaluation matrix across 5 different open-weight models (acting interchangeably as the Patient, Doctor, and Judge), we generated and evaluated extensive automated clinical interactions.

This intensive research phase was crucial to:
* **Benchmark Reasoning Limits:** Evaluate model performances across varying parameter sizes (from 3.8B to 9B) on complex medical traps.
* **Expose Critical LLM Flaws:** Detect and analyze behaviors such as compliance bias (sycophancy), persona drift, and medical hallucinations.
* **Determine Optimal Pairings:** Mathematically identify the safest and most rigorous model combinations to deploy in the final production UI.

---

## 🚀 Installation & Setup

### 1. Environment Preparation

We highly recommend using a dedicated Conda environment to avoid dependency conflicts, especially regarding PyTorch and CUDA versions.

```bash
conda create -n medsim_env python=3.11
conda activate medsim_env
```

### 2. Install Dependencies

Install the required packages. The `requirements.txt` is pre-configured to download the PyTorch version compatible with CUDA 12.1 (optimized for V100/Sigma2 nodes).

```bash
pip install -r requirements.txt
```

---

## ⚙️ How to Use MedSim

### Phase 1: Data Engineering (ETL)

To respect GitHub file size limits and copyright distributions, the full database of 9,500+ pathologies is **not included** in this repository.
To build the complete RAG database locally, run the scraper. It will use `pubmed-statpearls-set.txt` as a seed to query the NCBI database.

```bash
python etl/web_scrapping.py
```

*⚠️ **Note:** This process respects server rate limits and will take several hours to complete. For immediate testing, the repository includes a lightweight `knowledge_base_extract.json`.*

### Phase 2: The Interactive UI (Streamlit)

To launch the interactive clinical simulator designed for human medical students:

```bash
streamlit run app.py
```

This boots the dual-agent environment, featuring dynamic VRAM swapping to fit heavily quantized 7B-9B parameter models onto a single 16GB GPU.

### Phase 3: The MLOps Evaluation Factory

To objectively determine which LLM makes the best pedagogical judge, the system includes a fully automated benchmarking pipeline. It evaluates models like Meta Llama-3, Google Gemma-2, Mistral, BioMistral, and Phi-3.

Run the flagship algorithms in the following order:

1. **Generate the interactions:**
```bash
python scripts/run_benchmark.py
```

2. **Extract and aggregate the JSON data into a CSV:**
```bash
python scripts/extract_results.py
```

3. **Generate metrological visualizations:**
```bash
python scripts/generate_graphs.py
```

You can view the final metrics, including sycophancy bias and clinical reasoning variance, inside the `/results/graphs/` directory.

---

## 👨‍💻 Authors & Acknowledgments

* **Arthur Prevel**
* **Aloïs Kamber**

Developed at **UiT - The Arctic University of Norway**.  
Data sourced ethically from the **NCBI StatPearls Database**.