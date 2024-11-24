from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import json

import csv
from typing import List
from langchain.schema import Document

load_dotenv()


def docs_to_json(documents: List[Document], output_file: str = None) -> str:
    """
    Convert LangChain documents to JSON format
    
    Args:
        documents: List of LangChain Document objects
        output_file: Optional file path to save JSON output
        
    Returns:
        JSON string representation of the documents
    """
    docs_list = []
    for doc in documents:
        doc_dict = {
            'page_content': doc.page_content,
            'metadata': doc.metadata
        }
        docs_list.append(doc_dict)
    
    json_str = json.dumps(docs_list, indent=2)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(json_str)
    
    return json_str

def docs_to_csv(documents: List[Document], output_file: str = None) -> List[List]:
    """
    Convert LangChain documents to CSV format
    
    Args:
        documents: List of LangChain Document objects
        output_file: Optional file path to save CSV output
        
    Returns:
        List of rows (as lists) containing the document data
    """
    # Create header row with metadata keys
    metadata_keys = set()
    for doc in documents:
        metadata_keys.update(doc.metadata.keys())
    
    headers = ['page_content'] + sorted(list(metadata_keys))
    rows = [headers]
    
    # Add document data
    for doc in documents:
        row = [doc.page_content]
        for key in headers[1:]:  # Skip page_content column
            row.append(doc.metadata.get(key, ''))
        rows.append(row)
    
    if output_file:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    return rows

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GEMINI_API_KEY"),model="models/text-embedding-004", task_type="retrieval_document")



# Load example document
with open("Cleaned_MOHAP.txt", encoding='utf-8') as f:
    MOHAP_data = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=128,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([MOHAP_data])

docs_to_json(texts,"ss.json")

text_content = []
urls = []
for text in texts:
    if text.page_content == "":
        texts.remove(text)
    elif text.page_content.startswith("URL: https:"):
        urls.append(text.page_content)
        texts.remove(text)
    else :
        text_content.append(text)

print("Before duplicate removal", len(texts))
print("Before duplicate removal", type(texts))

seen_set = set()
for text in texts:
    if text.page_content in seen_set:
        texts.remove(text)
    else:
        seen_set.add(text.page_content)





# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# index_name = "gemini-test"  

# index = pc.Index(index_name)
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)






# vector_store.add_documents(documents=texts, ids=[str(id) for id in range(len(text_content))])


