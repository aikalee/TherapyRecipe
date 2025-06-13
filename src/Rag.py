import json
import time
import faiss
import os
from dotenv import load_dotenv
import requests

import numpy as np

from sentence_transformers import SentenceTransformer
from together import Together


def load_json_to_db(file_path):
    with open(file_path) as f:
        db = json.load(f)
    return db

#------Embedding and FAISS Indexing Functions------
def make_embeddings(embedder, embedder_name,db):    
    """
    Make embeddings for the given database of chunks.
    """

    texts = [chunk['text'] for chunk in db]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    
    return embeddings


def save_embeddings(embedder_name, embeddings):    
    """
    Save embeddings to a .npy file.
    """
    file_path = os.path.join("data", "embeddings", f"{embedder_name.replace('/', '_')}.npy")
    np.save(file_path, embeddings)
    print(f"Saved embeddings for {embedder_name}...")
    
    
    
def load_embeddings(embedder_name):
    """
    Load embeddings from a .npy file.
    """
        # if embeddings already exist, load them, else make new embeddings
    try:
        file_path = os.path.join("data", "embeddings", f"{embedder_name.replace('/', '_')}.npy")
        embeddings = np.load(file_path, allow_pickle=True)
        print(f"Embeddings for {embedder_name} already exist. Loading them...")
    except FileNotFoundError:
        print(f"Embeddings for {embedder_name} not found. Making new embeddings...")
        embeddings = make_embeddings(embedder,embedder_name, db)
        save_embeddings(embedder_name, embeddings)
        
    
    return embeddings

def load_embedder_with_fallbacks(embedder_name):
    """
    Tries loading a SentenceTransformer model with multiple fallback strategies.
    Returns the loaded model if successful. Raises RuntimeError if all strategies fail.
    """
    strategies = [
        {"trust_remote_code": False, "device": "cpu", "description": "default sentence transformer", 'class': 'SentenceTransformer'},
        {"trust_remote_code": True,  "device": None, "description": "sentence transformer with trust_remote_code=True", 'class': 'SentenceTransformer'},
        {"description": "manual make transformer + pooling with sentenceTransformer", "class": "Manual"},
    ]

    for i, strategy in enumerate(strategies):
        try:
            print(f"[Attempt {i+1}] Loading embedder '{embedder_name}' with {strategy['description']}")
            
            if strategy["class"] == "SentenceTransformer":
                kwargs = {}
                if strategy.get("trust_remote_code"):
                    kwargs["trust_remote_code"] = True
                if strategy.get("device"):
                    kwargs["device"] = strategy["device"]
                model = SentenceTransformer(embedder_name, **kwargs)
            elif strategy["class"] == "Manual":
                word_embedding_model = models.Transformer(embedder_name)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            print(f"[Success] Loaded embedder with strategy: {strategy['description']}")
            return model
        
        except Exception as e:
            print(f"[Failure] '{strategy['description']}' failed: {e}")

    raise RuntimeError(f"All strategies failed to load embedder '{embedder_name}'.")

    
# --------------Faiss index functions-------------------
def build_faiss_index(embeddings):
    """
    Build a FAISS index using cosine similarity (via normalized inner product).
    """
    print("Building FAISS index (cosine similarity)...")
    
    # Step 1: Normalize embeddings to unit vectors (L2 norm = 1)
    faiss.normalize_L2(embeddings)
    
    # Step 2: Use inner product index (dot product == cosine after normalization)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    return index

def load_faiss_index(embedder_name, embeddings):
    """
    Load the FAISS index from a file.
    """
    
    try:
        
        # if file doesn't exist in the folder data/faiss_index, raise FileNotFoundError
        index_file = os.path.join("data", "faiss_index", f"{embedder_name.replace('/', '_')}_index.faiss")
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index file {index_file} not found.")
        
        print(f"FAISS index for {embedder_name} already exists. Loading it...")
        index = faiss.read_index(index_file)
        print(f"Loaded FAISS index from {index_file}.")
    except FileNotFoundError:
        print(f"FAISS index for {embedder_name} not found. Building new index...")
        index = build_faiss_index(embeddings)
        save_faiss_index(embedder_name, index)
        print(f"FAISS index for {embedder_name} built and saved.")
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
                "text": db[indices[0][i]]['text'],
                "section": db[indices[0][i]]['metadata']['section'],
                "chunk_id": db[indices[0][i]]['metadata']['chunk_id'],
            })
        # if this chunk has a referee_id, it is a table already, we don't need to add it again later
        if db[indices[0][i]]['metadata']['referee_id']:
            existed_tables.add(db[indices[0][i]]['metadata']['referee_id'])
            print
        try:
            if db[indices[0][i]]['metadata']['referenced_tables']:
                referenced_tables.update(db[indices[0][i]]['metadata']['referenced_tables'])
        except KeyError:
            continue

    #existed_tables = tables that already exist in the retrieved results
    #referenced_tables = tables that are referenced by chunks in the retrieved results
    #table_to_add = referenced_tables - existed_tables
    table_to_add = [table for table in referenced_tables if table not in existed_tables]
    
    print(f"existed tables: {existed_tables}")
    print(f"referenced tables: {referenced_tables}")
    print(f"Tables to add: {table_to_add}")

    # add the referenced tables in the db to the results if their referee_id is in table_to_add
    i = 0
    for chunk in referenced_tables_db:
        if chunk['metadata']['referee_id'] in table_to_add:
            results.append({
                "text": chunk['text'],
                "section": chunk['metadata']['section'],
                "chunk_id": chunk['metadata']['chunk_id'],
            })
            i += 1
        if i == len(table_to_add):
            break
    return results

def load_together_llm_client():
    """
    Load the Together LLM client with the provided API key.
    """
    load_dotenv()  # Load environment variables from .env file
    
    return Together(api_key=os.getenv("TOGETHER_API_KEY"))

# ---------- Prompt ----------
def construct_prompt(query, faiss_results):
    with open("src/system_prompt.txt", "r") as f:
        system_prompt = f.read().strip()

    prompt = f"""
### System Prompt
{system_prompt}

### User Query
{query}

### Clinical Guidelines Context
"""
    for res in faiss_results:
        prompt += f"- reference: {res['section']}\n- This paragraph is from section: {res['text']}\n"
    return prompt


# ===== new feature: memory =====
def construct_prompt_with_memory(query, faiss_results, chat_history=None, history_limit=4):
    print("=============Constructing prompt with memory===========")
    with open("src/system_prompt.txt", "r") as f:
        system_prompt = f.read().strip()

    prompt = f"### System Prompt\n{system_prompt}\n\n"

    if chat_history:
        prompt += "### Chat History\n"
        for m in chat_history[-history_limit:]:
            prompt += f"{m['role'].title()}: {m['content']}\n"
        prompt += "\n"

    prompt += f"### User Query\n{query}\n\n"
    prompt += "### Clinical Guidelines Context\n"
    for res in faiss_results:
        prompt += f"- reference: {res['section']}\n- This paragraph is from section: {res['text']}\n"
    return prompt


def call_llm(llm_client, prompt, stream_flag=False, max_tokens=500, temperature=0.05, top_p=0.9, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
    print(f"Calling LLM with model: {model_name}")
    print(f"With parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
    try:
        if stream_flag:
            # For streaming mode, return a generator
            def stream_generator():
                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )
                print("Streaming response received from API")
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield content
            return stream_generator()
        else:
            # For non-streaming mode, return content directly
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
            )
            content = response.choices[0].message.content
            return content
            
    except Exception as e:
        print("Error in call_llm:", str(e))
        print("Error type:", type(e))
        import traceback
        traceback.print_exc()
        raise

def call_ollama(prompt, model="mistral", stream_flag=False, max_tokens=500, temperature=0.05, top_p=0.9):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True
    }

    with requests.post(url, json=payload, stream=True) as response:
        print("===========Entering Ollama stream loop===========")
        for line in response.iter_lines():
            if line:
                try:
                    chunk = line.decode("utf-8")
                    data = json.loads(chunk)
                    yield data["response"]
                except Exception as e:
                    continue
    

def launch_depression_assistant(embedder_name, designated_client=None):
    """
    Launch the depression assistant with the loaded database and embeddings.
    """
    global db, referenced_tables_db, embedder, index, llm_client
    
    db = load_json_to_db("data/processed/guideline_db.json")
    referenced_tables_db = load_json_to_db("data/processed/referenced_table_chunks.json")

    embedder = load_embedder_with_fallbacks(embedder_name)
        
    print(f"Using embedder: {embedder_name}")
    
    embeddings = load_embeddings(embedder_name)
    index = load_faiss_index(embedder_name, embeddings)

    if designated_client is None:
        print("No LLM client provided. Loading Together LLM client...")
        try:
            llm_client = load_together_llm_client()
        except Exception as e:
            print("Failed to load Together LLM client. This might be related to user access. Please manually configure your LLM client API key.")
    else:
        print("Using provided LLM client.")
        llm_client = designated_client
    print("---------Depression Assistant is ready to use!--------------\n\n")
    

def depression_assistant(query, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", max_tokens=500, temperature=0.05, top_p=0.9, stream_flag=False, chat_history=None):   
    t1 = time.perf_counter()

    results = faiss_search(query, embedder, db, index, referenced_tables_db, k=3)
    t2 = time.perf_counter()
    print(f"[Time] FAISS search done in {t2 - t1:.2f} seconds.")
    
    #rerank the results to restore context logic order
    # don't think it works well so commenting it out for now
    # results = sorted(results, key=lambda x: x['chunk_id'] if 'chunk_id' in x else 0)

    prompt = construct_prompt_with_memory(query, results, chat_history=chat_history)
    t3 = time.perf_counter()

    if llm_client == "Run Ollama Locally":
        print(f"Running Ollama Locally with model: {model_name}, Make sure you have enough memory to run the model.")
        response = call_ollama(prompt, model_name, stream_flag, max_tokens=max_tokens, temperature=temperature, top_p=top_p,)
    else:
        response = call_llm(llm_client, prompt, stream_flag, max_tokens=max_tokens, temperature=temperature, top_p=top_p, model_name=model_name)
    t4 = time.perf_counter()
    
    print(f"[Time] LLM response took {t4 - t3:.2f} seconds.")
    print(f"[Total time] {t4 - t1:.2f} seconds for this query.\n\n")

    return results, response