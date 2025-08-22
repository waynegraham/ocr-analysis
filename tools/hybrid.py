import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer, util
import torch

# --- 1. Solr Connection Details ---
# IMPORTANT: This script assumes your Solr core has a 'vector' field indexed
# for dense vector search. You'll need to ensure the data is already there.
solr_url = "http://localhost:8983/solr"
solr_core = "trustees"
solr_endpoint = f"{solr_url}/{solr_core}/select"

# --- 2. Define the search term ---
query_text = "When did the Board first discuss tuition increases?"

# --- 3. Load the Sentence Embedding Model ---
print("Loading sentence embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.\n")

# --- 4. Encode the Query into a Dense Vector ---
print("Encoding query into a dense vector...")
query_embedding = model.encode(query_text, convert_to_tensor=False)
# Convert the tensor to a string for the Solr query
query_vector_str = np.array2string(query_embedding, separator=',', formatter={'float_kind':lambda x: "%.6f" % x})
query_vector_str = query_vector_str.replace('\n', '').replace(' ', '')
print("Encoding complete.\n")

# --- 5. Perform the Initial Dense Vector Search (Solr {!knn f=vector}) ---
# This constructs the Solr query to perform the dense vector search.
# We're using a filter query (fq) with the {!knn} syntax to get the top results.
# The 'vector' field in Solr should store the dense embeddings.
top_k_initial = 10
params = {
    'q': '{!knn f=vector topK=10}' + query_vector_str,
    'fl': 'id, text_field_to_rerank, score', # Fetch the ID, the text field to rerank, and the Solr score
    'rows': top_k_initial,
    'wt': 'json'
}

print(f"Executing initial dense vector search on Solr (topK={top_k_initial})...")
try:
    response = requests.get(solr_endpoint, params=params)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    solr_results = response.json()
    initial_top_results = solr_results['response']['docs']

    if not initial_top_results:
        print("No results found from the Solr query.")
    else:
        print("Initial Top Results from Solr (sorted by Solr's vector score):")
        for doc in initial_top_results:
            print(f"Solr Score: {doc.get('score', 'N/A'):.4f} | Doc: '{doc.get('text_field_to_rerank', 'No text found')}'")
        print("\n" + "="*50 + "\n")

except requests.exceptions.RequestException as e:
    print(f"Error connecting to Solr: {e}")
    initial_top_results = []

# --- 6. Rerank the Top-K Results ---
# We'll use the same reranking logic as before, but on the results from Solr.
print("Reranking the initial top results...")
if initial_top_results:
    reranked_results = []
    for doc in initial_top_results:
        document_text = doc.get('text_field_to_rerank', '')
        original_score = doc.get('score', 0)

        # Simple reranking logic based on keyword matching
        keyword_bonus = 0
        query_keywords = ["Board", "discuss", "tuition", "increases"]
        for keyword in query_keywords:
            if keyword.lower() in document_text.lower():
                keyword_bonus += 0.1 # Add a small bonus for each keyword match

        # The final rerank score is a combination of the two.
        rerank_score = original_score + keyword_bonus
        reranked_results.append({
            'id': doc.get('id'),
            'text': document_text,
            'original_score': original_score,
            'rerank_score': rerank_score
        })

    # Sort the results again based on the new rerank score.
    reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

    # --- 7. Print the Final Reranked Results ---
    print(f"Final Reranked Top {len(initial_top_results)} Results:")
    for result in reranked_results:
        print(
            f"Rerank Score: {result['rerank_score']:.4f} "
            f"(Original: {result['original_score']:.4f}) | "
            f"Doc: '{result['text']}'"
        )
else:
    print("Skipping reranking due to no results from Solr.")