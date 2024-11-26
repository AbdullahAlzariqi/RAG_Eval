
from openai import OpenAI

from dotenv import load_dotenv
from typing import List, Dict

#Can be moved to main.py
import os 
load_dotenv()




class openai_generator:
    def __init__(self):
        self.model = OpenAI()
        self.model.api_key = os.getenv("OPENAI_API_KEY") 
    
    def generate(self, query, chunks):

        chunk_combined = ""
        for chunk in chunks:
            chunk_combined += chunk + "\n"
        prompt = f"You are a useful agent. You will answer this query: {query} by using these chunks:{chunk_combined}"
        res = self.model.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
                )
        
        return res.choices[0].message.content
    

