import requests
import re
import json
import logging

# Setup logging
logging.basicConfig(filename="solr_queries.log", level=logging.INFO, format="%(message)s")


# Function to clean and strip irrelevant words from text corpora
def clean_text(text):
    # Define a basic stopword list or import from libraries like NLTK
    stopwords = set(['the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'for', 'of', 'with', 'to', 'as', 'in', 'by'])

    # Tokenize the text (simple split by space here, you can use NLP libraries for better tokenization)
    words = re.findall(r'\b\w+\b', text.lower())

    # Remove stopwords and return the cleaned text
    cleaned_words = [word for word in words if word not in stopwords]

    return cleaned_words


# Function to create Solr queries from cleaned text
def create_solr_queries(cleaned_words, field="body"):
    # Join cleaned words into a Solr query string using "OR"
    query = " OR ".join([f"{field}:{word}" for word in cleaned_words])
    return query


# Function to execute Solr query and capture debug information
def execute_solr_query(solr_url, query):
    # Add Solr query parameters for debug
    params = {
        'q': query,
        'debugQuery': 'true',
        'debug.explain.structured': 'false',
        'indent': 'true',
        'q.op': 'OR'
    }

    # Send a GET request to Solr
    response = requests.get(solr_url, params=params)

    # Check if the response is successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error executing query: {query}")
        return None


# Function to process Solr's debug information
def process_debug_info(solr_response):
    if 'debug' in solr_response:
        debug_info = solr_response['debug']
        query_log = {
            "rawquerystring": debug_info.get("rawquerystring", ""),
            "querystring": debug_info.get("querystring", ""),
            "parsedquery": debug_info.get("parsedquery", ""),
            "parsedquery_toString": debug_info.get("parsedquery_toString", "")
        }
        # Log this info for training the GPT-2 model
        logging.info(json.dumps(query_log))


# Main function to run the pipeline
def main():
    # Example text corpora
    text_corpora = [
        "This is a simple query to test the huggingface model functionality.",
        "Another example of how solr works with lucene queries."
    ]

    # Solr collection URL (change this to your actual Solr collection URL)
    solr_url = "http://localhost:8983/solr/huggingface/select"

    for text in text_corpora:
        # Step 1: Clean the text corpora
        cleaned_words = clean_text(text)

        # Step 2: Create Solr queries from the cleaned text
        solr_query = create_solr_queries(cleaned_words)
        print(f"Generated Solr Query: {solr_query}")

        # Step 3: Execute the Solr query
        solr_response = execute_solr_query(solr_url, solr_query)

        # Step 4: Process and log debug information
        if solr_response:
            process_debug_info(solr_response)


if __name__ == "__main__":
    main()
