import google.generativeai as genai
import os
import dotenv
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class gemini_retriever:
    
    def get_embeddings(self,query):
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_document",
            title="Embedding of single string")["embedding"]
        return result
    
    def get_Chunks(self,query, k=10):
        embedding = self.get_embeddings(query)
        
        pc = Pinecone(os.getenv("PINECONE_API_KEY"))

        index = pc.Index("gemini-test")

        matches = index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True
        )["matches"]
        # ids = []
        # for match in matches:
        #     ids.append(match["id"])
        return matches

            
        