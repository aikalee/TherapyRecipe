import json
import time
import faiss
import os
from dotenv import load_dotenv

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
    Build a FAISS index for the given embeddings.
    """    
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(embeddings)  # Add embeddings to the index
    
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
        if db[indices[0][i]]['metadata']['referenced_tables']:
            referenced_tables.update(db[indices[0][i]]['metadata']['referenced_tables'])

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


def call_llm(llm_client, prompt, stream_flag=False, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
    try:
        if stream_flag:
            # For streaming mode, return a generator
            def stream_generator():
                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.05,
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
                max_tokens=500,
                temperature=0.05,
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
        llm_client = load_together_llm_client()
    else:
        print("Using provided LLM client.")
        llm_client = designated_client
    print("---------Depression Assistant is ready to use!--------------\n\n")
    

def depression_assistant(query, stream_flag=False):    
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

    
    response = call_llm(llm_client, prompt, stream_flag)

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

def write_batched_results(embedder_name, result_path):
    
    time0 = time.perf_counter()
    launch_depression_assistant(embedder_name)
    print(f"[Time] Launching Depression Assistant took {time.perf_counter() - time0:.2f} seconds.")
    
    queries, answers = load_queries_and_answers("data/raw/queries.txt", "data/raw/answers.txt")

    # write results into 2 file, 
    # Response by {embedder_name} Embedder and LLama3.3 70B
    # Retrieved Results by {embedder_name} Embedder
    
    embedder_filename = embedder_name.replace('/', '_')
    

    with open(f"{result_path}Retrieved_Results_by_{embedder_filename}.md", "w") as f1, \
        open(f"{result_path}Response_by_{embedder_filename}.md", "w") as f2:

        for i, query in enumerate(queries):
            result, response = depression_assistant(query)

            # Write retrieved results
            f1.write(f"## Query {i+1}\n")
            f1.write(f"{query.strip()}\n\n")
            f1.write("## Answer\n")
            f1.write(f"{answers[i].strip()}\n\n")
            f1.write("## Retrieved Results\n")
            
            for res in result:
                f1.write(f"\n\n#### {res['section']}\n\n")
                f1.write(f"{res['text']}\n")
            f1.write("\n\n---\n\n")

            # Write response
            f2.write(f"## Query {i+1}\n")
            f2.write(f"{query.strip()}\n\n")
            f2.write("## Answer\n")
            f2.write(f"{answers[i].strip()}\n\n")
            
            f2.write(f"## Response\n")
            f2.write(response)
            f2.write("\n\n---\n\n")
            break


if __name__ == "__main__":
    embedder_name = "allenai/longformer-base-4096"
    # embedder_name = "emilyalsentzer/Bio_ClinicalBERT"
    # embedder_name = "Qwen/Qwen3-Embedding-0.6B"
    # embedder_name = "all-MiniLM-L6-v2"
    # embedder_name = "jinaai/jina-embeddings-v3"
    # embedder_name = "abhinand/MedEmbed-large-v0.1"
    # embedder_name = "BAAI/bge-base-en-v1.5",
    # embedder_name = "BAAI/bge-large-en-v1.5"
    # embedder_name = "BAAI/bge-small-en-v1.5"
    # embedder_name = "intfloat/multilingual-e5-base"
    # embedder_name = "sentence-transformers/all-mpnet-base-v2"
    # embedder_name = 'pritamdeka/S-PubMedBert-MS-MARCO',
    # embedder_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    # embedder_name = 'all-MiniLM-L6-v2'
    
    result_path = "data/results/week_5_generation/"
    
    write_batched_results(embedder_name, result_path)