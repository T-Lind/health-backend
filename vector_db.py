from astrapy import DataAPIClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env", override=True)

client = OpenAI()

class AstraDBVectorStore:
    """
    Class to abstract functionality for vector db in AstraDB.
    AstraDB is a leader in vector-search databases and the team really knows what they're doing (I worked on it).
    They stay up-to-date with the latest research and have a great team of engineers.
    Plus, Cassandra (CQL) just has a lot of great advantages built in.
    """

    def __init__(self, token, api_endpoint, namespace, collection_name, dimension):
        self.client = DataAPIClient(token)
        self.database = self.client.get_database_by_api_endpoint(api_endpoint, namespace=namespace)
        try:
            self.collection = self.database.get_collection(collection_name)
        except:
            self.collection = self.database.create_collection(collection_name, dimension=dimension)
    def insert_vector(self, metadata, vector):
        document = metadata.copy()
        document["$vector"] = vector
        self.collection.insert_one(document)

    def insert_vectors(self, metadata, vectors):
        documents = []
        for vector in vectors:
            document = metadata.copy()
            document["$vector"] = vector
            documents.append(document)
        self.collection.insert_many(documents)

    def search_vectors(self, query_vector, limit=10):
        results = self.collection.find(
            filter={},  # Metadata filter, not used in this case b/c I'm searching through all vectors
            vector=query_vector,
            limit=limit,
            include_similarity=True,
        )
        # TODO: get the whole interaction from the id
        refined_results = []
        for result in results:
            refined_results.append({
                "similarity": result["$similarity"],
                "content": result["content"],
                "context": result["context"],
            })

        return refined_results

    def search_query(self, query: str, limit=10):
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-large",
        )
        emb = response.data[0].embedding
        responses = self.search_vectors(emb, limit=limit)

        return responses
