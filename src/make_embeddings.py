from sentence_transformers import SentenceTransformer
import json
import numpy as np

filename='data/processed/guideline_db_with_img.json'
embedder_name = 'all-MiniLM-L6-v2'


# read in json file
with open(filename, "r", encoding="utf-8") as f:
    db = json.load(f)

embedder = SentenceTransformer(embedder_name)

texts = [entry["text"] for entry in db]
embeddings = embedder.encode(texts, convert_to_numpy=True)


np.save(embedder_name+".npy", embeddings)
print("###"*20)
print(f"Embeddings shape{embeddings.shape} saved to {embedder_name}.npy")