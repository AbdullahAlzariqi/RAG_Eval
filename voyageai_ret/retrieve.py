import os
import dotenv
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
import voyageai

load_dotenv()





class voyage_retriever:

    def _get_embeddings(self, query):
        vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        result = vo.embed([query], model="voyage-3", input_type="document")
        return result.embeddings[0]
    
    def get_Chunks(self,query):
        embedding = self._get_embeddings(query)

        pc = Pinecone(os.getenv("PINECONE_API_KEY"))

        index = pc.Index("voyageai-test")

        matches = index.query(
            vector=embedding,
            top_k=10,
            include_metadata=True
        )["matches"]
        # ids = []
        # for match in matches:
        #     ids.append(match["id"])
        return matches


