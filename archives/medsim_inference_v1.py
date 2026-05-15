import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import os

# ==========================================
# 1. ENGINE CONFIGURATION (V100 16GB)
# ==========================================
model_id = "BioMistral/BioMistral-7B-DARE"

# 4-bit configuration for 16GB VRAM compatibility
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print(f"🚀 Loading BioMistral (DARE version)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# ==========================================
# 2. KNOWLEDGE BASE MANAGEMENT (JSON)
# ==========================================
def get_pathology_info(search_term, file_path='knowledge_base_extract.json'):
    """ Searches for a specific pathology in the JSON file """
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Search for a match in titles (case-insensitive)
    for item in data:
        if search_term.lower() in item['title'].lower():
            return item
    return None

# ==========================================
# 3. INFERENCE ENGINE (RAG)
# ==========================================
def ask_medsim(pathology_name, question):
    # Retrieve data from JSON
    record = get_pathology_info(pathology_name)
    
    if not record:
        return f"Sorry, I have no information regarding '{pathology_name}'."

    # Prepare context using relevant sections
    useful_sections = ["History and Physical", "Evaluation", "Treatment / Management"]
    context = ""
    for title, content in record['sections'].items():
        if title in useful_sections:
            context += f"### {title}\n{content}\n\n"

    # Construct the prompt (Mistral special format [INST] ... [/INST])
    prompt = f"""[INST] You are a medical expert. Answer the question based exclusively on the provided context.

CONTEXT:
{context}

QUESTION:
{question} [/INST]"""

    # Encoding and sending to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print(f"🧠 Analysis in progress for: {pathology_name}...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=400, 
            temperature=0.2,          # High precision
            repetition_penalty=1.2,  # Prevent loops/repetition
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decoding the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the AI's response (after the [/INST] tag)
    return full_response.split("[/INST]")[-1].strip()

# ==========================================
# 4. QUICK TEST ZONE
# ==========================================
if __name__ == "__main__":
    # Test: Clinical Diagnosis
    pathology = "Osteoarthritis"
    test_question = "What are the risk factors and clinical signs?"
    
    print("\n" + "="*50)
    response = ask_medsim(pathology, test_question)
    print(f"\nBIOMISTRAL RESPONSE:\n{response}")
    print("="*50 + "\n")
