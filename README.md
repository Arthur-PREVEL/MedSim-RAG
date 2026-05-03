# MedSim-RAG: Realistic Patient Simulation for Medical Training[cite: 1]

**MedSim-RAG** is an advanced AI simulation platform designed for medical students to practice diagnostic interviewing[cite: 1]. Developed as part of a **Master’s in Health Engineering**, it leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to create a safe, accurate, and interactive learning environment[cite: 1].

## 🌟 Key Features[cite: 1]

- **Dual-Agent Architecture**:[cite: 1]
  - **The Patient (Jean)**: A 70-year-old persona that describes symptoms in plain language, avoiding medical jargon to simulate a real-life clinical encounter[cite: 1].
  - **The Supervisor**: An AI evaluator that reviews the conversation transcript against gold-standard clinical data to provide feedback and a grade[cite: 1].
- **Clinical Grounding (RAG)**: Powered by a database of **9,500+ pathologies** (StatPearls), ensuring responses are strictly based on verified medical knowledge[cite: 1].
- **Optimized for HPC**: Specifically tuned for **NVIDIA Tesla V100 (16GB)** hardware using 4-bit NF4 quantization to maximize performance without exceeding VRAM limits[cite: 1].
- **Advanced Decoding**: Implements `repetition_penalty` and `no_repeat_ngram_size` to ensure coherent, non-cyclic medical dialogues[cite: 1].

## 🛠️ Technical Stack[cite: 1]

- **Model**: [BioMistral-7B-DARE](https://huggingface.co/BioMistral/BioMistral-7B-DARE)[cite: 1]
- **Frameworks**: `Transformers`, `BitsAndBytes`, `Accelerate`, `PyTorch`[cite: 1]
- **Inference**: 4-bit NF4 Quantization & SDPA (Scaled Dot Product Attention)[cite: 1]
- **Environment**: Python 3.11 / Conda[cite: 1]

## 🚀 Getting Started[cite: 1]

### 1. Environment Setup[cite: 1]
Since the project is optimized for CUDA 12.1, it is recommended to use Conda:[cite: 1]
```bash
conda create -n medsim_env python=3.11 -y
conda activate medsim_env
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers accelerate bitsandbytes sentencepiece
```[cite: 1]

### 2. Knowledge Base[cite: 1]
Ensure your `knowledge_base_clean.json` (containing the 9,500 pathologies) is in the root directory[cite: 1].

### 3. Run the Simulation[cite: 1]
Launch the interactive patient simulation:[cite: 1]
```bash
python medsim_patient.py
```[cite: 1]

## 📊 Project Architecture[cite: 1]

1. **Retrieval**: System fetches relevant clinical data from the JSON database[cite: 1].
2. **Context Injection**: Medical facts are formatted into a "System Prompt" to guide the LLM's persona[cite: 1].
3. **Inference**: The model generates responses in character as a patient[cite: 1].
4. **Evaluation**: Upon exit, the Supervisor agent analyzes the exchange and provides a pedagogical debrief[cite: 1].

---
*Developed for the Master in Health Engineering Evaluation - April 2026.*[cite: 1]
