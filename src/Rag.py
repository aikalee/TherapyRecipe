import json
import time
import torch
import faiss
import os
from dotenv import load_dotenv

import numpy as np

from pydantic import BaseModel
from typing import Optional, List
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from together import Together


# ---------- data structure ----------
class Metadata(BaseModel):
    section: str
    type: str
    chunk_id: Optional[int] = None
    headings: str
    referee_id: Optional[str] = None
    referenced_tables: Optional[List[str]] = None


class Chunk(BaseModel):
    text: str
    metadata: Metadata


# ---------- data load----------
def load_json_to_db(file_path):
    with open(file_path) as f:
        db_raw = json.load(f)
    db = [Chunk(**chunk) for chunk in db_raw]
    return db


# ---------- optional custom TransformerEmbedder ----------
class TransformerEmbedder:
    def __init__(self, model_name, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

    def encode(self, texts, batch_size=8, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model(**encoded)
            embeddings = self.mean_pooling(output, encoded['attention_mask'])
            all_embeddings.append(embeddings.cpu())

        embs = torch.cat(all_embeddings, dim=0)
        return embs.numpy() if convert_to_numpy else embs


# ---------- Embedding & save ----------
def make_embeddings(embedder, embedder_name, db):
    texts = [chunk.text for chunk in db]
    return embedder.encode(texts, convert_to_numpy=True)


def save_embeddings(embedder_name, db):
    embeddings = make_embeddings(embedder_name, db)
    file_path = os.path.join("data", "embeddings", f"{embedder_name.replace('/', '_')}.npy")
    np.save(file_path, embeddings)


def load_embeddings(embedder_name):
    file_path = os.path.join("data", "embeddings", f"{embedder_name.replace('/', '_')}.npy")
    return np.load(file_path, allow_pickle=True)


# ---------- FAISS ----------
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def load_faiss_index(embedder_name):
    index_file = os.path.join("data", "faiss_index", f"{embedder_name.replace('/', '_')}_index.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError
    return faiss.read_index(index_file)


def save_faiss_index(embedder_name, index):
    index_file = os.path.join("data", "faiss_index", f"{embedder_name.replace('/', '_')}_index.faiss")
    faiss.write_index(index, index_file)


# ---------- search ----------
def faiss_search(query, embedder, db, index, referenced_table_db, k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    results, referenced_tables, existed_tables = [], set(), set()

    for i in range(k):
        if indices[0][i] == -1:
            continue
        chunk = db[indices[0][i]]
        results.append(
            {"text": chunk.text, "section": chunk.metadata.section, "chunk_id": chunk.metadata.chunk_id}
        )
        if chunk.metadata.referee_id:
            existed_tables.add(chunk.metadata.referee_id)
        if chunk.metadata.referenced_tables:
            referenced_tables.update(chunk.metadata.referenced_tables)

    table_to_add = [t for t in referenced_tables if t not in existed_tables]
    for chunk in referenced_table_db:
        if chunk.metadata.referee_id in table_to_add:
            results.append(
                {"text": chunk.text, "section": chunk.metadata.section, "chunk_id": chunk.metadata.chunk_id}
            )
            if len(table_to_add) == len(results):
                break
    return results


# ---------- LLM ----------
def load_together_llm_client():
    load_dotenv()
    return Together(api_key=os.getenv("TOGETHER_API_KEY"))


def call_llm(llm_client, prompt, stream_flag=False, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.05,
        stream=stream_flag,
    )
    if stream_flag:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    else:
        return response.choices[0].message.content


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
def construct_prompt_with_memory(query, faiss_results, chat_history=None, history_limit=3):
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


# ---------- Assistant ----------
def launch_depression_assistant(embedder_name="all-MiniLM-L6-v2", designated_client=None):
    global db, referenced_tables_db, embedder, index, llm_client

    db = load_json_to_db("data/processed/guideline_db.json")
    referenced_tables_db = load_json_to_db("data/processed/referenced_table_chunks.json")

    try:
        embedder = SentenceTransformer(embedder_name)
    except Exception:
        embedder = SentenceTransformer(embedder_name, trust_remote_code=True, device='cpu')

    try:
        embeddings = load_embeddings(embedder_name)
    except FileNotFoundError:
        embeddings = make_embeddings(embedder, embedder_name, db)
        save_embeddings(embedder_name, db)

    try:
        index = load_faiss_index(embedder_name)
    except FileNotFoundError:
        index = build_faiss_index(embeddings)
        save_faiss_index(embedder_name, index)

    llm_client = designated_client or load_together_llm_client()
    print("---------Depression Assistant is ready!---------\n")


def depression_assistant(query, stream_flag=False, chat_history=None):
    t0 = time.perf_counter()

    results = faiss_search(query, embedder, db, index, referenced_tables_db, k=3)
    prompt = construct_prompt_with_memory(query, results, chat_history)

    resp = call_llm(llm_client, prompt, stream_flag)

    if stream_flag:
        for chunk in resp:
            yield chunk
    else:
        return results, resp

    print(f"[Total] {(time.perf_counter() - t0):.2f}s\n")
