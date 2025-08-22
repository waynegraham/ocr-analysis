# main.py
# This script demonstrates how to perform a dense vector search query in Solr 9.x
# using a sentence encoded by the 'all-MiniLM-L6-v2' model.

# Ensure you have the required libraries installed:
# pip install requests sentence-transformers

import requests
import json
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# URL for your Solr core's select handler
SOLR_URL = "http://localhost:8983/solr/trustees/select"

# The name of the dense vector field in your Solr schema.
# IMPORTANT: You must replace 'my_vector_field' with the actual name of your field.
VECTOR_FIELD = "vector"

# The sentence you want to find similar documents for.
QUERY_TEXT = "When was campus exapansion first discussed"

# The sentence-transformer model to use for encoding the query text.
# This should be the same model used to generate the vectors in your index.
MODEL_NAME = 'all-MiniLM-L6-v2'

# The number of top results to return.
TOP_K = 10

# --- Main Script ---

def perform_vector_search():
    """
    Encodes a query, sends it to Solr for a dense vector search,
    and prints the results.
    """
    print(f"Loading sentence transformer model: '{MODEL_NAME}'...")
    try:
        # 1. Load the sentence transformer model
        model = SentenceTransformer(MODEL_NAME)

        # 2. Encode the query text into a vector
        print(f"Encoding query: '{QUERY_TEXT}'")
        query_vector = model.encode(QUERY_TEXT).tolist()
        
        # The vector must be a standard Python list for JSON serialization.
        # The 'all-MiniLM-L6-v2' model produces a 384-dimension vector.
        print(f"Successfully generated a {len(query_vector)}-dimension vector.")

        # 3. Construct the JSON payload for the Solr query
        # The query uses the {!knn} query parser for K-Nearest Neighbor search.
        # - 'f' specifies the vector field to search against.
        # - 'topK' is the number of similar documents to retrieve.
        # The vector itself is passed as a JSON array string.
        
        # Convert the vector list to a comma-separated string without spaces.
        query_vector_str = "[" + ",".join(map(str, query_vector)) + "]"

        # FIX: Restructure the payload for Solr's JSON Request API.
        # The error "Unknown top-level key...: q" means that 'q' is not a valid
        # top-level key. Instead, we use 'query', 'limit', and 'fields'.
        solr_query_payload = {
            "query": f"{{!knn f={VECTOR_FIELD} topK={TOP_K}}}{query_vector_str}",
            "limit": TOP_K,
            "fields": ["id", "score"]  # 'fl' becomes 'fields' and is a list
        }

        # 4. Post the query to Solr
        headers = {'Content-type': 'application/json'}
        print(f"\nSending query to Solr at: {SOLR_URL}")
        print("Payload being sent:")
        print(json.dumps(solr_query_payload, indent=2))


        response = requests.post(
            SOLR_URL,
            data=json.dumps(solr_query_payload),
            headers=headers
        )

        # Raise an exception if the request returned an error status code
        response.raise_for_status()

        # 5. Print the results
        print("\n--- Solr Response ---")
        results = response.json()
        print(json.dumps(results, indent=2))
        print("---------------------\n")
        
        num_found = results.get('response', {}).get('numFound', 0)
        if num_found > 0:
            print(f"Query successful. Found {num_found} documents.")
        else:
            print("Query successful, but no documents were found.")


    except requests.exceptions.RequestException as e:
        print(f"\n--- ERROR ---")
        print(f"Failed to connect to Solr or the query was malformed.")
        print(f"Please check if Solr is running and the URL is correct.")
        print(f"Details: {e}")
        # If you get a 400 Bad Request, check the Solr logs for more details on the parsing error.
        print("-------------")
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(f"Details: {e}")
        print("------------------------------------")


if __name__ == "__main__":
    # Before running, make sure to update the VECTOR_FIELD variable!
    if VECTOR_FIELD == "vector":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: Please update the 'VECTOR_FIELD' variable in   !!!")
        print("!!!          this script with the name of your dense vector !!!")
        print("!!!          field in your Solr schema.                     !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    
    perform_vector_search()
