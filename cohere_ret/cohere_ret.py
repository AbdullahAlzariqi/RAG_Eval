from pinecone.grpc import PineconeGRPC as Pinecone
import cohere
from dotenv import load_dotenv
from typing import List, Dict

#Can be moved to main.py
import os 
load_dotenv()


class cohere_retriever:

    def __init__(self,db_name = "cohere-test"):
        self.db_name = db_name

    def get_embeddings(self, query):
        co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

        embedding = co.embed(
            texts=[query],
            model="embed-multilingual-v3.0",
            input_type="search_query",
            embedding_types=["float"],
             ).embeddings.float_[0]
        return embedding
    
    def get_Chunks(self,query,k=10):
        embedding = self.get_embeddings(query)

        pc = Pinecone(os.getenv("PINECONE_API_KEY"))

        index = pc.Index(self.db_name)

        matches = index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True
        )["matches"]
        # ids = []
        # for match in matches:
        #     ids.append(match["id"])
        return matches

       




















