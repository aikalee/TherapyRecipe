import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import faiss

faiss.omp_set_num_threads(1)



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


def load_json_to_db(file_path: str) -> List[Chunk]:
    """
    Load JSON file into a list of Chunk objects.
    """
    with open(file_path) as f:
        raw = json.load(f)
    return [Chunk(**chunk) for chunk in raw]


class TransformerEmbedder:
    def __init__(self, model_name: str, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)

    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            emb = self.mean_pooling(out, enc["attention_mask"])
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0).numpy()


def make_embeddings(embedder: TransformerEmbedder, db: List[Chunk]) -> np.ndarray:
    texts = [chunk.text for chunk in db]
    return embedder.encode(texts)


def save_embeddings(embedder_name: str, embeddings: np.ndarray):
    path = os.path.join("data", "embeddings", f"{embedder_name.replace('/', '_')}.npy")
    np.save(path, embeddings)
    print(f"Saved embeddings for {embedder_name}")


def load_embeddings(embedder_name: str) -> np.ndarray:
    path = os.path.join("data", "embeddings", f"{embedder_name.replace('/', '_')}.npy")
    return np.load(path, allow_pickle=True)


def load_embedder_with_fallbacks(model_name: str) -> SentenceTransformer:
    strategies = [
        {"trust_remote_code": False, "device": "cpu"},
        {"trust_remote_code": True, "device": None},
    ]
    for strat in strategies:
        try:
            kwargs = {}
            if strat.get("trust_remote_code"): kwargs["trust_remote_code"] = True
            if strat.get("device"): kwargs["device"] = strat["device"]
            return SentenceTransformer(model_name, **kwargs)
        except Exception:
            continue
    # Manual fallback
    from sentence_transformers import models
    w = models.Transformer(model_name)
    p = models.Pooling(w.get_word_embedding_dimension())
    return SentenceTransformer(modules=[w, p])


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings)
    return idx


def save_faiss_index(embedder_name: str, index: faiss.IndexFlatL2):
    path = os.path.join("data", "faiss_index", f"{embedder_name.replace('/', '_')}_index.faiss")
    faiss.write_index(index, path)
    print(f"Saved FAISS index to {path}")


def load_faiss_index(embedder_name: str) -> faiss.IndexFlatL2:
    path = os.path.join("data", "faiss_index", f"{embedder_name.replace('/', '_')}_index.faiss")
    return faiss.read_index(path)


def faiss_search(query: str, embedder: TransformerEmbedder, db: List[Chunk], index: faiss.IndexFlatL2, referenced_table_db: List[Chunk], k: int =3) -> List[dict]:
    q_emb = embedder.encode([query])
    distances, inds = index.search(q_emb, k)
    results, ref_tables, exist_tables = [], set(), set()
    for i in range(k):
        idx = inds[0][i]
        if idx == -1: continue
        chunk = db[idx]
        results.append({
            "text": chunk.text,
            "section": chunk.metadata.section,
            "chunk_id": chunk.metadata.chunk_id,
        })
        ref_id = getattr(chunk.metadata, "referee_id", None)
        if ref_id: exist_tables.add(ref_id)
        rts = getattr(chunk.metadata, "referenced_tables", []) or []
        ref_tables.update(rts)
    to_add = [t for t in ref_tables if t not in exist_tables]
    for ch in referenced_table_db:
        if getattr(ch.metadata, "referee_id", None) in to_add:
            results.append({
                "text": ch.text,
                "section": ch.metadata.section,
                "chunk_id": ch.metadata.chunk_id,
            })
            if len(results) >= k + len(to_add): break
    return results


def load_together_llm_client() -> Together:
    load_dotenv()
    return Together(api_key=os.getenv("TOGETHER_API_KEY"))


def call_llm(llm_client: Together, prompt: str, stream_flag: bool=False, max_tokens: int=500, temperature: float=0.05, top_p: float=0.9, model_name: str="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
    resp = llm_client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream_flag
    )
    if stream_flag:
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    else:
        return resp.choices[0].message.content


def construct_prompt(query: str, faiss_results: List[dict]) -> str:
    with open("src/system_prompt.txt") as f:
        sp = f.read().strip()
    p = f"""
### System Prompt
{sp}

### User Query
{query}

### Clinical Guidelines Context
"""
    for r in faiss_results:
        p += f"- reference: {r['section']}\n- {r['text']}\n"
    return p


def construct_prompt_with_memory(query: str, faiss_results: List[dict], chat_history: list=None, history_limit: int=4) -> str:
    with open("src/system_prompt.txt") as f:
        sp = f.read().strip()
    p = f"### System Prompt\n{sp}\n\n"
    if chat_history:
        p += "### Chat History\n"
        for m in chat_history[-history_limit:]:
            p += f"{m['role'].title()}: {m['content']}\n"
        p += "\n"
    p += f"### User Query\n{query}\n\n### Clinical Guidelines Context\n"
    for r in faiss_results:
        p += f"- reference: {r['section']}\n- {r['text']}\n"
    return p


def launch_depression_assistant(embedder_name: str, designated_client=None):
    global db, referenced_table_db, embedder, index, llm_client
    db = load_json_to_db("data/processed/guideline_db.json")
    referenced_table_db = load_json_to_db("data/processed/referenced_table_chunks.json")
    embedder = load_embedder_with_fallbacks(embedder_name)
    embeddings = load_embeddings(embedder_name)
    try:
        index = load_faiss_index(embedder_name)
    except FileNotFoundError:
        index = build_faiss_index(embeddings)
        save_faiss_index(embedder_name, index)
    llm_client = designated_client or load_together_llm_client()
    print("Depression Assistant is ready.")


def depression_assistant(query: str, stream_flag: bool=False, max_tokens: int=500, temperature: float=0.05, top_p: float=0.9, model_name: str="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", chat_history: list=None):
    results = faiss_search(query, embedder, db, index, referenced_table_db)
    prompt = construct_prompt_with_memory(query, results, chat_history)
    return prompt, call_llm(llm_client, prompt, stream_flag, max_tokens, temperature, top_p, model_name)
