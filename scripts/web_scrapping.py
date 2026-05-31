import requests
from bs4 import BeautifulSoup
import time
import json
import re

def extract_ids_from_file(path):
    """Extracts NBK IDs from the provided text file using regex."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Keeping your original regex pattern
    return re.findall(r'AID - (NBK\d+)', content)

def extract_validated_data(nbk_id):
    """Scrapes and validates clinical data for a given NBK ID."""
    url = f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: 
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # --- TITLE CORRECTION (Anti-Bookshelf logic) ---
        # We retrieve all h1 tags and ignore the generic "Bookshelf" title
        page_title = nbk_id  # Default value
        for h1 in soup.find_all('h1'):
            txt = h1.get_text(strip=True)
            if txt.lower() != "bookshelf":
                page_title = txt
                break
        
        # --- SECTION LOGIC (H2 & Siblings) ---
        h2_titles = soup.find_all('h2')
        sections = {}
        for t in h2_titles:
            title_text = t.get_text(strip=True)
            content = []
            curr = t.find_next_sibling()
            
            while curr and curr.name != 'h2':
                if curr.name in ['p', 'ul', 'ol']:
                    content.append(curr.get_text(separator=' ', strip=True))
                curr = curr.find_next_sibling()
            
            if content:
                sections[title_text] = " ".join(content)

        # --- CRITICAL FILTER ---
        # We only keep records that contain a "History and Physical" section
        if "History and Physical" in sections:
            return {
                "id": nbk_id,
                "title": page_title,
                "sections": sections
            }
        return None 
    except Exception:
        return None

# --- MAIN EXECUTION ---
ids = extract_ids_from_file('data/pubmed-statpearls-set.txt')
knowledge_base = []

print(f" Analysis started: {len(ids)} IDs detected.")

for i, nbk in enumerate(ids):
    data = extract_validated_data(nbk)
    if data:
        knowledge_base.append(data)
        print(f"[{i+1}/{len(ids)}] VALIDATED: {data['title']}")
    else:
        # We remain discrete about skipped records to keep the console clean
        if (i+1) % 10 == 0:
            print(f"[{i+1}/{len(ids)}]... processing ...")
    
    # Safety backup every 50 validated articles
    if len(knowledge_base) % 50 == 0 and data:
        with open('knowledge_base_clean.json', 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=4)

    time.sleep(1.1)  # Respect the server (Rate limiting)

# Final save
with open('knowledge_base_clean.json', 'w', encoding='utf-8') as f:
    json.dump(knowledge_base, f, ensure_ascii=False, indent=4)

print(f"\n Finished, {len(knowledge_base)} clinical records ready.")
