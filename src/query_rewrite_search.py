import pickle

import json
from sentence_transformers import SentenceTransformer

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from together import Together
llm_client = Together(api_key='4f6e44b7689d6592b2b5b57ad3940ac9f488d14c22802e8bcdf641b06e98cbbe')
#4f6e44b7689d6592b2b5b57ad3940ac9f488d14c22802e8bcdf641b06e98cbbe

with open("../data/processed/guideline_db_with_img.json") as f:
    db = json.load(f)
    
with open('../data/processed/Exp2_MedEmb.emb', mode='rb') as f: #replace with your file
  vector_store = pickle.load(f)



def depression_assistant(query):
    embedder = SentenceTransformer('abhinand/MedEmbed-large-v0.1')
    print("--------- We're using the MedEmbed-large-v0.1 model for embeddings.---------")
    
    original_query_results = search(embedder, query, vector_store, k=4, min_similarity=0.3)
    print(f"Original query: {query}")
    
    new_queries = rewrite_query(query)
    new_queries_results = []
    
    for new_query in new_queries:
        if new_query.lower().endswith("none**.") or new_query.lower().endswith("none**"):
            print(f"------- :Rewritten query: {new_query}")
            print("------- :No relevant information found in the query, skipping search.")
            new_queries_results.append([None])
            continue
        print(f"------- :Rewritten query: {new_query}")
        results = search(embedder, new_query, vector_store, k=4, min_similarity=0.3)
        print(f"------- :Results number: {len(results)}")
        # print(f"------- :Results: {results}")
        print()
        new_queries_results.append(results)
        
    prompt = construct_prompt(query, original_query_results, new_queries, new_queries_results)
    response = call_llm(prompt)
    return response

def search(embedder, query, vector_store, k, min_similarity):
    query_embedding = embedder.encode(query.lower())

    similarities = []
    results = []
    
    referenced_tables = list()
    # calculate cosine similarity between each text and the query
    for i, chunk in enumerate(vector_store):
        similarity = cosine_similarity([query_embedding], [chunk["embedding"]])
        if similarity[0][0] >= min_similarity:
            similarities.append((i, similarity[0][0]))

    # sort the similarities based on similarity and select the top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    for i, similarity in similarities[:k]:
            results.append({'text':db[i]['text'], 'section': db[i]['metadata']['section'], 'type': db[i]['metadata'].get('referee_id', 'paragraph')})
            try:
                for table in db[i]["metadata"]["referenced_tables"]:
                    # check if the table is already in the set
                    table = table.lower().replace(" ", "_").replace(".", "_")
                    referenced_tables.append(table)
            except KeyError:
                # if there is no referenced tables, Means this is a table, skip
                pass
            
    referenced_tables = set(referenced_tables)  # remove duplicates
    print(referenced_tables)
    
    
    for chunk in results:
        #if table is already in the results, skip
        try:
            if chunk["type"] in referenced_tables:
                referenced_tables.remove(chunk["type"])
                print(f"Removed table: {chunk["type"]}")
                print(referenced_tables)
        except KeyError:
            # if there is no referee_id, skip
            pass
        

    for chunk in db:
        try:
            if chunk["metadata"]["referee_id"] in referenced_tables:
                results.append({'text': chunk['text'],'section': chunk['metadata']['section'], 'type': chunk['metadata']['referee_id']})
                print(f"Added table: {chunk['metadata']['referee_id']}")
        except KeyError:
            # if there is no referee_id, skip
            pass
    
    if not results or not results[0]:
        return ["No matching documents!"]
    return results


def construct_prompt(query, original_query_results, new_queries, new_queries_results):
    system_prompt = (
        "Your name is Depression Assistant, a helpful and friendly recipe assistant. "
        "Summarize the clinical guidelines provided in the context and then tried to answer the user query. "
        "If the query or guideline provided is not related to depression, please say 'I am not sure about that'. Don't make up things. "
        
    )

    prompt = f"""
    ### System Prompt
    {system_prompt}

    ### User Query
    {query}
    ### clinical guidelines details
    {json.dumps(original_query_results, indent=2)}
    """
    
    for i, new_result in enumerate(new_queries_results):
        if new_result[0]:
            prompt += f"""
            ### Rewritten Query {i+1}
            {new_queries[i]}
            ### clinical guidelines details
            {json.dumps(new_result, indent=2)}
            """
    
    return prompt

def rewrite_query(query):
    system_prompt = (
        """extract info from the user query to answer the following question.

            question:

            - Is the patient currently in the acute or maintenance phase of depression treatment, and what symptoms are present?

            - Has the patient received pharmacotherapy, psychotherapy, or a combination of both?

            - What specific antidepressant medications have been administered so far?

            - Is the patient experiencing any side effects or adverse reactions to the current medication?

            - Has the patient's condition improved, remained the same, or worsened under the current treatment plan?

            make the keyword in your answer bold font,Don't return anything else: Answer each question in a new line, and if the question is not applicable, write "none" in that line.
            """)
    
    prompt = f"""
        ### System Prompt
        {system_prompt}
        ### User Query
        {query}
        """
    response = call_llm(prompt)
    # clean the response
    response = response.split("\n")
    new_queries = [line.strip() for line in response if line.strip()]
    # print("this is the re-written query")
    # print(response)
    
    return new_queries

def call_llm(prompt):

    response = llm_client.chat.completions.create(
      model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", #don't change the model!
      messages=[
          {
              "role": "user",
              "content": prompt
          }
      ],
      max_tokens=500
    )
    return response.choices[0].message.content

#main function to run the depression assistant
if __name__ == "__main__":

    print("Welcome to the Depression Assistant!")
    print("This assistant will help you find relevant information from the clinical guidelines.")
    print("You can enter your query and the assistant will try to answer it based on the clinical guidelines.")
    print("If you want to exit, just type 'exit' or 'quit'.")
    #run the depression assistant until the user decides to stop
    query = input("Please enter your query: ")
    if not query:
        print("No query provided. If you want to exit, type 'exit'.")
    while query.lower() not in ['exit']:
        #if the query is not empty, run the depression assistant
        if query.strip():
            response = depression_assistant(query)
            #while we're still processing the response, tell the user we're working on it
            print("Processing your query, please wait...")
            print(f"------------------This is the response from the Depression Assistant: {response}-------------------")
        else:
            print("No query provided. If you want to exit, type 'exit'.")
        query = input("Please enter your query: ")