import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import time

# ==========================================
# 1. OPTIMIZED CONFIGURATION (V100 16GB)
# ==========================================
model_id = "BioMistral/BioMistral-7B-DARE"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("🚀 Loading BioMistral V2...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Using attn_implementation="sdpa" to accelerate inference on V100 GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa" 
)

# Pre-loading the database (to save time on every question)
print("📂 Loading Knowledge Base (Extract)...")
with open('knowledge_base_extract.json', 'r', encoding='utf-8') as f:
    KNOWLEDGE_BASE = json.load(f)

# ==========================================
# 2. IMPROVED INFERENCE LOGIC
# ==========================================
def ask_medsim_v2(pathology_name, question):
    # 1. Fast search in cache
    record = next((item for item in KNOWLEDGE_BASE if pathology_name.lower() in item['title'].lower()), None)
    
    if not record:
        return "Pathology not found in database."

    # 2. Context Construction
    # We focus on the most relevant clinical sections
    context = " ".join([f"{k}: {v}" for k, v in record['sections'].items() if k in ["History and Physical", "Evaluation"]])
    
    prompt = f"[INST] You are a medical expert. Provide a concise and accurate response based on the context.\nCONTEXT: {context}\nQUESTION: {question} [/INST]"

    # 3. Optimized Generation
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    start_time = time.time()
    
    with torch.inference_mode(): # Faster than no_grad()
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.1,         # Keeps the model factual
            repetition_penalty=1.2,  # Prevents repetitive loops
            no_repeat_ngram_size=3,  # Prevents "stuttering" issues
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    inference_time = time.time() - start_time
    print(f"⚡ Inference completed in {inference_time:.2f}s")
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

# ==========================================
# 3. PERFORMANCE TEST
# ==========================================
if __name__ == "__main__":
    patho = "Osteoarthritis"
    question = "What are the major risk factors?"
    
    result = ask_medsim_v2(patho, question)
    print(f"\n[BIOMISTRAL V2 RESPONSE]:\n{result}")
