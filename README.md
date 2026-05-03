# MedSim-RAG: Realistic Patient Simulation for Medical Training[cite:

**MedSim-RAG** is an advanced AI simulation platform designed for medical students to practice diagnostic interviewing. Developed as part of a **Master’s in Health Engineering**, it leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to create a safe, accurate, and interactive learning environment.

## 🌟 Key Features

- **Dual-Agent Architecture**:
  - **The Patient (Jean)**: A 70-year-old persona that describes symptoms in plain language, avoiding medical jargon to simulate a real-life clinical encounter.
  - **The Supervisor**: An AI evaluator that reviews the conversation transcript against gold-standard clinical data to provide feedback and a grade.
- **Clinical Grounding (RAG)**: Powered by a database of **9,500+ pathologies** (StatPearls), ensuring responses are strictly based on verified medical knowledge.
- **Optimized for HPC**: Specifically tuned for **NVIDIA Tesla V100 (16GB)** hardware using 4-bit NF4 quantization to maximize performance without exceeding VRAM limits.
- **Advanced Decoding**: Implements `repetition_penalty` and `no_repeat_ngram_size` to ensure coherent, non-cyclic medical dialogues.

## 🛠️ Technical Stack

- **Model**: [BioMistral-7B-DARE](https://huggingface.co/BioMistral/BioMistral-7B-DARE)
- **Frameworks**: `Transformers`, `BitsAndBytes`, `Accelerate`, `PyTorch`
- **Inference**: 4-bit NF4 Quantization & SDPA (Scaled Dot Product Attention)
- **Environment**: Python 3.11 / Conda

## 📂 Data Information & Acquisition

To keep this repository lightweight and respect storage limits, the provided `knowledge_base_extract.json` contains only **4 sample clinical records** for testing purposes. 

To utilize the full simulation potential with the complete database of **9,500+ pathologies**, you must generate the dataset locally:

1. Ensure you have the `pubmed-statpearls-set.txt` file in your directory.
2. Run the acquisition script:
   ```bash
   python web_scrapping.py

Note: The scraping process respects server *rate limits* and may take several hours to complete the full *9,500+* record set.

## 🚀 Getting Started

### 1. Environment Setup
Since the project is optimized for CUDA 12.1, it is recommended to use Conda:
```bash
conda create -n medsim_env python=3.11 -y
conda activate medsim_env
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers accelerate bitsandbytes sentencepiece
```

### 2. Knowledge Base
Ensure your `knowledge_base_clean.json` (containing the 9,500 pathologies) is in the root directory.

### 3. Run the Simulation
Launch the interactive patient simulation:
```bash
python medsim_patient.py
```

## 📊 Project Architecture

1. **Retrieval**: System fetches relevant clinical data from the JSON database.
2. **Context Injection**: Medical facts are formatted into a "System Prompt" to guide the LLM's persona.
3. **Inference**: The model generates responses in character as a patient.
4. **Evaluation**: Upon exit, the Supervisor agent analyzes the exchange and provides a pedagogical debrief.

---
*Developed for the Master in Health Engineering Evaluation INF3600 - UiT - April-May 2026.*
