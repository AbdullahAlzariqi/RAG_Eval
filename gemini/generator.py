
import google.generativeai as genai
import os 
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class gemini_generator:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash-002")

    
    def generate(self, query, chunks):
        chunk_combined = ""
        for chunk in chunks:
            chunk_combined += chunk + "\n"
        prompt = f"You are a useful agent. You will answer this query: {query} by using these chunks:{chunk_combined}"
        res = self.model.generate_content(prompt).text
        return res