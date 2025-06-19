# Therapy Recipe - Depression Assistant Chatbot (RAG System)

A Retrieval-Augmented Generation (RAG) chatbot specialized in answering questions about clinical depression guidelines, featuring multiple embedding model and LLM provider support.

## Features

- **Medical RAG Pipeline**: Specialized for CANMAT depression guidelines
- **Multi-Embedding vector_store cached**: vector_store preloaded with 12 medical/general embedding models
- **LLM Provider Flexibility**: TogetherAI/OpenAI/Ollama/NVIDIA options
- **Dynamic Table Handling**: Automatic retrieval of referenced tables
- **Streaming Responses**: Real-time answer generation
- **Parameter Control**: Adjustable temp, top_p, max_length

## Tech Stack

### Core Components
- **Backend**: Python 3.12
- **Frontend**: Streamlit
- **Embeddings**: Sentence Transformers
- **Vector DB**: FAISS
- **LLMs**: Llama-3.3-70B (default), configurable alternatives

## Repository Structure
```
.
├── data
│   ├── embeddings/       # Saved embedding arrays (.npy)
│   ├── faiss_index/      # FAISS index files
│   ├── processed/        # Processed JSON databases
│   └── raw/              # Original guideline documents
├── documentation/        # Project docs and reports
├── evaluation/           # evaluation data, scripts and results
└── src
    ├── Rag.py            # Core RAG pipeline
    ├── app.py            # Streamlit frontend
    ├── data_processing/  # Document preprocessing
    └── run_batched_queries/ # Execution multiple queries and write to a markdown file
```

## Installation

1. Clone repository:
   ```bash
   git clone 
   cd TherapyRecipe
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## CLI Batch Processing
```bash
python src/run_batched_queries/write_result_to_file.py embedding_model_name output_path
```

## Usage

### Web Interface
```bash
streamlit run src/app.py
```

## Evaluation

Benchmarking scripts in `evaluation/` evaluate:
- Retrieval accuracy on different Embedder performance
- Hallucination of Generation
- Whole System's quality
