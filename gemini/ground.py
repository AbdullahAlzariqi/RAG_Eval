from trulens.providers.langchain import Langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from trulens.feedback import GroundTruthAgreement
import numpy as np
from trulens.core import Feedback
from trulens.core import Select
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone.grpc import PineconeGRPC as Pinecone





load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("gemini-test")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
embed = genai.embed_content
embed_model_name = "models/text-embedding-004"
task_type="retrieval_document"


class retriever:
        def __init__(self, embed, embed_model_name, index):
             self.embed = embed
             self.embed_model_name = embed_model_name
             self.index = index
        def get_data(self,query):
            embedding=self.embed(
            model=self.embed_model_name,
            task_type=task_type,
            content= query)["embedding"]

            vecs = self.index.query(
            vector=embedding,
            top_k=5,
            includeMetadata=True,
            include_values=True
        )["matches"]
            ids=[] 
            for match in vecs:
                ids.append(match.id)
            data = self.index.fetch(ids)
            docs = []
            for key in data["vectors"]:
                docs.append(data["vectors"][key]["metadata"]["text"])
            return docs



ret = retriever(embed=embed, embed_model_name="models/text-embedding-004", index=index)
chunks = ret.get_data("How long does it take to get a controlled drugs prescription book?")
retrieved_chunks_formatted = retrieved_chunks_formatted = [{"text": chunk, "score": 1} for chunk in chunks[:4]] 




llm = ChatGoogleGenerativeAI(google_api_key=os.getenv("GEMINI_API_KEY"), model="gemini-1.5-pro")

golden_set = [
    {
        "query": "benefits of exercise",
        "expected_response": "Health better it is",

         "expected_chunks": [
            {"text": "Improved cardiovascular health", "expect_score": 1},
            {"text": "Better mood", "expect_score": 2},
            {"text": "Weight management", "expect_score": 3},
            {"text": "Stronger bones", "expect_score": 4},
        ]
    }
]


ground_truth_collection = GroundTruthAgreement(golden_set, provider=Langchain(chain = llm))



ncdg = ground_truth_collection.ndcg_at_k(
    query= "benefits of exercise",
    retrieved_context_chunks= retrieved_chunks_formatted
)

print(ncdg)
print("\n" )
print(chunks)

