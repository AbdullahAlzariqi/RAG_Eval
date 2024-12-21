from utils.chunk_scorer import score_chunk
import json

def prepare_expected_chunks(json_file_path, output_file_path):
   with open(json_file_path, 'r') as f:
       data = json.load(f)
   
   expected_chunks = data['expected_chunks']
   expected_responses = data['expected_response']
   
   processed_chunks = []
   for i, chunks in enumerate(expected_chunks):
       chunk_dicts = [
           {
               "text": chunk,
               "title": chunk,
               "expected_score": score_chunk(chunk, expected_responses[i])
           }
           for chunk in chunks
       ]
       processed_chunks.append(chunk_dicts)
   
   # Replace original expected_chunks with processed ones
   data['expected_chunks'] = processed_chunks
   
   # Write modified data to new json file
   with open(output_file_path, 'w') as f:
       json.dump(data, f, indent=4)

# Use function
