# MedSim-RAG: Realistic Patient Simulation for Medical Training

**MedSim-RAG** is an advanced AI simulation platform designed for medical students to practice diagnostic interviewing. Developed as part of a **Master’s in Health Engineering**, it leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to create a safe, accurate, and interactive learning environment.

## Key Features

- **Dual-Agent Architecture**: 
  - **The Patient (Jean)**: A 70-year-old persona that describes symptoms in plain language, avoiding medical jargon to simulate a real-life clinical encounter.
  - **The Supervisor**: An AI evaluator that reviews the conversation transcript against gold-standard clinical data to provide feedback and a grade.
- **Clinical Grounding (RAG)**: Powered by a database of **9,500+ pathologies** (StatPearls), ensuring responses are strictly based on verified medical knowledge.
- **Optimized for HPC**: Specifically tuned for **NVIDIA Tesla V100 (16GB)** hardware using 4-bit NF4 quantization to maximize performance without exceeding VRAM limits.
- **Advanced Decoding**: Implements `repetition_penalty` and `no_repeat_ngram_size` to ensure coherent, non-cyclic medical dialogues.

## Technical Stack

- **Model**: [BioMistral-7B-DARE](https://huggingface.co/BioMistral/BioMistral-7B-DARE)
- **Frameworks**: `Transformers`, `BitsAndBytes`, `Accelerate`, `PyTorch`
- **Inference**: 4-bit NF4 Quantization & SDPA (Scaled Dot Product Attention)
- **Environment**: Python 3.11 / Conda

## Getting Started

### 1. Environment Setup
Since the project is optimized for CUDA 12.1, it is recommended to use Conda:

```bash
conda create -n medsim_env python=3.11 -y
conda activate medsim_env
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers accelerate bitsandbytes sentencepiece
