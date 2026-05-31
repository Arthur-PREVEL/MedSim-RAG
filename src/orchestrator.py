import json
import os

class MedSimOrchestrator:
    """
    Handles data partitioning for the Multi-Agent Simulation.
    Ensures an Information Firewall between the Patient, Doctor, and Judge.
    """
    def __init__(self, file_path=None):
        # UPDATED: Dynamically point to the data directory relative to this file
        if file_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(base_dir, 'data', 'knowledge_base_extract.json')
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found.")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.db = json.load(f)

    def get_context(self, pathology_name, role):
        record = next((item for item in self.db if pathology_name.lower() in item['title'].lower()), None)
        if not record:
            return None

        if role == "patient":
            # Patient only knows their symptoms
            return record['sections'].get('History and Physical', "General discomfort.")
        
        elif role == "judge":
            # VRAM OPTIMIZATION: We only give the Judge the Title and the Symptoms.
            # Sending the whole JSON file causes a Context Window OOM on 16GB GPUs.
            symptoms = record['sections'].get('History and Physical', "No symptoms listed.")
            return f"DIAGNOSIS: {record['title']}\n### KEY SYMPTOMS TO CHECK FOR:\n{symptoms}"
        
        elif role == "doctor":
            # Doctor is blind to the record
            return "General medical knowledge."
        
        return None