import csv
from collections import defaultdict
from dotenv import load_dotenv
import os
from openai import OpenAI
from vector_db import AstraDBVectorStore

# Load environment variables
load_dotenv(".env", override=True)

# Initialize OpenAI client
client = OpenAI()

# Initialize AstraDBVectorStore
astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
astra_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
astra_namespace = "example_convos"
collection_name = "exchanges"
dimension = 3072

vector_store = AstraDBVectorStore(astra_token, astra_api_endpoint, astra_namespace, collection_name, dimension)

def read_csv_and_remove_duplicates(file_path):
    unique_contexts = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            context = row['Context']
            response = row['Response']
            unique_contexts[context].append(response)
    return unique_contexts

def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large",
    )
    return response.data[0].embedding

def main():
    csv_file_path = "data/exhanges.csv"
    unique_contexts = read_csv_and_remove_duplicates(csv_file_path)

    for context, responses in unique_contexts.items():
        vector = embed_text(context)
        metadata = {
            "content": responses[0],  # Use the first response if there are multiple
            "context": context
        }
        vector_store.insert_vector(metadata, vector)
        print(f"Inserted vector for context: {context[:50]}...")

if __name__ == "__main__":
    main()