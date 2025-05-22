import json

with open('../data/processed/guideline_db.json', 'r', encoding='utf-8') as f1:
    text = json.load(f1)

with open('../data/processed/tables.json', 'r', encoding='utf-8') as f2:
    tables = json.load(f2)

combined = text + tables

with open('../data/processed/guideline_db.json', 'w', encoding='utf-8') as out:
    json.dump(combined, out, indent=4, ensure_ascii=False)
