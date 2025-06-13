from Rag import launch_depression_assistant, depression_assistant

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