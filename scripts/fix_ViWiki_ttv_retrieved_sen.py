import json
import os

# Load both files
with open('/data/npl/ICEK/ICECAP/icecap-/ViWiki_data/ViWiki_ttv_old.json', 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

with open('', 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

# Create a mapping from filename to paragraphs in file2
file2_mapping = {}
for entry in data2.values():
    filename = os.path.basename(entry["image_path"])
    file2_mapping[filename] = entry["paragraphs"]

# Replace retrieved_sentences in data1
for item in data1:
    img_name = item.get("img_name")  # like "0000009544.jpg"
    if img_name in file2_mapping:
        item["retrieved_sentences"] = file2_mapping[img_name]

# Save updated file1 if needed
with open('/data/npl/ICEK/ICECAP/icecap-/ViWiki_data/ViWiki_ttv.json', 'w', encoding='utf-8') as f_out:
    json.dump(data1, f_out, ensure_ascii=False, indent=2)
