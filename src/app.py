from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

embedder_name = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(embedder_name)

class UserQuery(BaseModel):
    question: str


# load data from json file and load embeddings from npy file
def load_data():
    # Load documents from a JSON file
    import json
    with open('data/guideline_db.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    # Load embeddings from a numpy file
    embeddings = np.load(embedder_name + ".npy")
    return documents, embeddings

documents, doc_embeddings = load_data()



@app.post("/ask")
async def answer_question(query: UserQuery):
    # 1. convert user query to embedding
    query_embedding = embedder.encode([query.question])[0]
    
    # 2. do similarity search
    scores = np.dot(doc_embeddings, query_embedding)
    
    # 3. get the top 3 relevant documents's index and context
    top_indices = np.argsort(scores)[-3:][::-1]
    top_contexts = [documents[i]['text'] for i in top_indices]
    top_index = top_indices[0]
    
    return {"answer": top_contexts, "source_id": top_index}
    
