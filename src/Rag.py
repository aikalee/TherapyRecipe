import json
import time
import faiss
import os
from dotenv import load_dotenv

import numpy as np

from pydantic import BaseModel
from typing import Optional, List, Union
from sentence_transformers import SentenceTransformer

from together import Together



class Metadata(BaseModel):
    section: str
    type: str
    chunk_id: Optional[int]= None
    headings: str
    referee_id: Optional[str] = None
    referenced_tables: Optional[List[str]] = None

class Chunk(BaseModel):
    text: str
    metadata: Metadata


def load_json_to_db(file_path):
    with open(file_path) as f:
        db_raw = json.load(f)
    db = [Chunk(**chunk) for chunk in db_raw]
    return db


#------Embedding and FAISS Indexing Functions------
def make_embeddings(embedder_name,db):    
    """
    Make embeddings for the given database of chunks.
    """
    embedder = SentenceTransformer(embedder_name)

    texts = [chunk.text for chunk in db]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    
    return embeddings


def save_embeddings(embedder_name, db):    
    """
    Save embeddings to a .npy file.
    """
    embeddings = make_embeddings(embedder_name, db)
    
    print(f"Saving embeddings for {embedder_name}...")
    file_path = os.path.join("data", "embeddings", f"{embedder_name.replace('/', '_')}.npy")
    np.save(file_path, embeddings)
    
    
def load_embeddings(embedder_name):
    """
    Load embeddings from a .npy file.
    """
    print(f"Loading embeddings for {embedder_name}...")
    file_path = os.path.join("data", "embeddings", f"{embedder_name.replace('/', '_')}.npy")
    embeddings = np.load(file_path, allow_pickle=True)
    
    return embeddings
    
# --------------Faiss index functions-------------------
def build_faiss_index(embeddings):
    """
    Build a FAISS index for the given embeddings.
    """    
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(embeddings)  # Add embeddings to the index
    
    return index

def load_faiss_index(embedder_name):
    """
    Load the FAISS index from a file.
    """
    index_file = f"{embedder_name.replace('/', '_')}_index.faiss"
    
    # if file doesn't exist in the folder data/faiss_index, raise FileNotFoundError
    index_file = os.path.join("data", "faiss_index", index_file)
    
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index file {index_file} not found.")
    
    index = faiss.read_index(index_file)
    print(f"Loaded FAISS index from {index_file}.")
    return index

def save_faiss_index(embedder_name, index):
    """
    Save the FAISS index to a file.
    """
    index_file = f"{embedder_name.replace('/', '_')}_index.faiss"
    
    index_file = os.path.join("data", "faiss_index", index_file)
    
    faiss.write_index(index, index_file)
    print(f"Saved FAISS index to {index_file}.")

# ---------------------------------

def faiss_search(query, embedder, db, index,referenced_table_db, k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    results = []
    referenced_tables = set()
    existed_tables = set()
    for i in range(k):
        if indices[0][i] != -1:  # Check if the index is valid
            results.append({
                "text": db[indices[0][i]].text,
                "section": db[indices[0][i]].metadata.section,
                "chunk_id": db[indices[0][i]].metadata.chunk_id,
            })
        # if this chunk has a referee_id, it is a table already, we don't need to add it again later
        if db[indices[0][i]].metadata.referee_id:
            existed_tables.add(db[indices[0][i]].metadata.referee_id)
        if db[indices[0][i]].metadata.referenced_tables:
            referenced_tables.update(db[indices[0][i]].metadata.referenced_tables)

        #perform .lower().replace(" ", "_").replace(".", "_") to all the table in the referenced_tables
    table_to_add = {table.lower().replace(" ", "_").replace(".", "_") for table in referenced_tables if table not in existed_tables}

    # add the referenced tables in the db to the results if their referee_id is in table_to_add
    i = 0
    for chunk in referenced_tables_db:
        if chunk.metadata.referee_id in table_to_add:
            results.append({
                "text": chunk.text,
                "section": chunk.metadata.section,
                "chunk_id": chunk.metadata.chunk_id,
            })
            i += 1
        if i == len(table_to_add):
            break
    return results

def load_together_llm_client(api_key):
    """
    Load the Together LLM client with the provided API key.
    """
    load_dotenv()  # Load environment variables from .env file
    
    return Together(api_key=os.getenv('TOGETHER_API_KEY', api_key))

def call_llm(llm_client, prompt, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
        temperature=0.05
    )
    return response.choices[0].message.content



def construct_prompt(query, faiss_results):
    # reads system prompt from a file
    with open("src/system_prompt.txt", "r") as f:
        system_prompt = f.read().strip()

    prompt = f"""
### System Prompt
{system_prompt}

### User Query
{query}

### Clinical Guidelines Context
    """
    for result in faiss_results:
        prompt += f"- reference: {result['section']}\n- This paragraph is from section: {result['text']}\n"

    return prompt

    

def launch_depression_assistant(embedder_name="all-MiniLM-L6-v2"):
    """
    Launch the depression assistant with the loaded database and embeddings.
    """
    global db, referenced_tables_db,  embedder, index, llm_client
    
    db = load_json_to_db("data/processed/guideline_db.json")
    referenced_tables_db = load_json_to_db("data/processed/referenced_table_chunks.json")
    embedder = SentenceTransformer(embedder_name)
    print(f"Using embedder: {embedder_name}")
    
    # if embeddings already exist, load them, else make new embeddings
    try:
        embeddings = load_embeddings(embedder_name)
        print(f"Embeddings for {embedder_name} already exist. Loading them...")
    except FileNotFoundError:
        print(f"Embeddings for {embedder_name} not found. Making new embeddings...")
        embeddings = make_embeddings(embedder_name, db)
        save_embeddings(embedder_name, db)
    
    try:
        index = load_faiss_index(embedder_name)
        print(f"FAISS index for {embedder_name} already exists. Loading it...")
    except FileNotFoundError:
        print(f"FAISS index for {embedder_name} not found. Building new index...")
        index = build_faiss_index(embeddings)
        save_faiss_index(embedder_name, index)
        print(f"FAISS index for {embedder_name} built and saved.")
    
    llm_client = load_together_llm_client(api_key='4f6e44b7689d6592b2b5b57ad3940ac9f488d14c22802e8bcdf641b06e98cbbe')
    print("---------Depression Assistant is ready to use!--------------\n\n")
    

def depression_assistant(query):    
    t1 = time.perf_counter()

    results = faiss_search(query, embedder, db, index, referenced_tables_db, k=3)
    t2 = time.perf_counter()
    print(f"[Time] FAISS search done in {t2 - t1:.2f} seconds.")
    
    #rerank the results to restore context logic order
    # don't think it works well so commenting it out for now
    # results = sorted(results, key=lambda x: x['chunk_id'] if 'chunk_id' in x else 0)

    prompt = construct_prompt(query, results)
    t3 = time.perf_counter()
    print(f"[Time] Prompt construction took {t3 - t2:.2f} seconds.")

    response = call_llm(llm_client, prompt)
    t4 = time.perf_counter()
    print(f"[Time] LLM response took {t4 - t3:.2f} seconds.")

    print(f"[Total time] {t4 - t1:.2f} seconds for this query.\n\n")
    return results, response

def load_queries_and_answers(query_file, answers_file):
    """
    Load queries and answers from the provided files.
    """
    with open(query_file, 'r') as f:
        queries = f.readlines()
    
    with open(answers_file, 'r') as f:
        answers = f.readlines()
    
    return queries, answers

def main():
    # if we want to use a different embedder, change this variable
    embedder_name = "all-MiniLM-L6-v2"
    launch_depression_assistant(embedder_name)
    
    queries, answers = load_queries_and_answers("data/raw/queries.txt", "data/raw/answers.txt")

    with open(f"{embedder_name.replace("/", "_")}_llama3.3_70B.md", "w") as f:
        for i, query in enumerate(queries):
            result, response = depression_assistant(query)
            # write the response to a md file
            f.write(f"## Query {i+1}\n")
            f.write(f"{query.strip()}\n\n")
            f.write("#### Answer\n")
            f.write(f"{answers[i].strip()}\n\n")
            f.write("#### Retrieved Results\n")
            for res in result:
                f.write(f"\n\n- Section: {res['section']}\n")
                f.write(f"  Text: {res['text']}\n")
            f.write(f"#### Response by {embedder_name} Embedder and LLama3.3 70B\n")
            f.write(response.strip())
            f.write("\n\n---\n\n")
            break

if __name__ == "__main__":
    main()