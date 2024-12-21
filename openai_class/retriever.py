from pinecone.grpc import PineconeGRPC as Pinecone
from openai import OpenAI

from dotenv import load_dotenv
from typing import List, Dict

#Can be moved to main.py
import os 
load_dotenv()



class openai_retriever:

    def get_embeddings(self, query):
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        embedding = client.embeddings.create(
    input=query,
    model="text-embedding-3-large"
    ).data[0].embedding

        return embedding
    
    def get_Chunks(self,query,k=10):
        embedding = self.get_embeddings(query)

        pc = Pinecone(os.getenv("PINECONE_API_KEY"))

        index = pc.Index("openai-test")

        matches = index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True
        )["matches"]
        # ids = []
        # for match in matches:
        #     ids.append(match["id"])
        return matches

       




















