import cohere
from dotenv import load_dotenv
import os 
load_dotenv()


class cohere_generator:
    def __init__(self):
        self.co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
    
    def generate(self, query, chunks):
        chunk_combined = ""
        for chunk in chunks:
            chunk_combined += chunk+ "\n"
        prompt = f"You are a useful agent. You will answer this query: {query} by using these chunks:{chunk_combined}"
        response = self.co.chat(
            model="command-r-08-2024",
            messages=[{"role": "user", "content": prompt}],
            )
        return response.message.content[0].text
