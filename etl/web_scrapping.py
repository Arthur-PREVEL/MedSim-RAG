import requests
from bs4 import BeautifulSoup
import time
import json
import re

def extract_ids_from_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Keep the original regex
    return re.findall(r'AID - (NBK\d+)', content)

def extract_validated_data(nbk_id):
    url = f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # --- TITLE CORRECTION (Anti-Bookshelf) ---
        # Retrieve all h1 tags and ignore the one named "Bookshelf"
        page_title = nbk_id # Default value
        for h1 in soup.find_all('h1'):
            txt = h1.get_text(strip=True)
            if txt.lower() != "bookshelf":
                page_title = txt
                break
        
        # --- SECTION LOGIC (H2 & Siblings) ---
        h2_titles = soup.find_all('h2')
        sections = {}
        for t in h2_titles:
            title_txt = t.get_text(strip=True)
            content_list = []
            curr = t.find_next_sibling()
            while curr and curr.name != 'h2':
                if curr.name in ['p', 'ul', 'ol']:
                    content_list.append(curr.get_text(separator=' ', strip=True))
                curr = curr.find_next_sibling()
            if content_list:
                sections[title_txt] = " ".join(content_list)

        # --- CRITICAL FILTER ---
        # Keep only records that have a clinical history section
        if "History and Physical" in sections:
            return {
                "id": nbk_id,
                "title": page_title, # Use the corrected title here
                "sections": sections
            }
        return None 
    except:
        return None

# --- GLOBAL EXECUTION ---
# Adjusted paths based on the new directory structure (/etl and /data)
input_file = '../data/pubmed-statpearls-set.txt'
output_file = '../data/knowledge_base_clean.json'

ids = extract_ids_from_file(input_file)
knowledge_base = []

print(f"Starting analysis: {len(ids)} IDs detected.")

for i, nbk in enumerate(ids):
    data = extract_validated_data(nbk)
    if data:
        knowledge_base.append(data)
        print(f"[{i+1}/{len(ids)}] VALIDATED: {data['title']}")
    else:
        # Stay discreet on discarded items to avoid polluting the console
        if (i+1) % 10 == 0:
            print(f"[{i+1}/{len(ids)}]... in progress ...")
    
    # Security save every 50 validated articles
    if len(knowledge_base) % 50 == 0 and data:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=4)

    time.sleep(1.1) # Respect the server rate limit

# Final save
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(knowledge_base, f, ensure_ascii=False, indent=4)

print(f"\n Mission accomplished! {len(knowledge_base)} clinical records ready.")