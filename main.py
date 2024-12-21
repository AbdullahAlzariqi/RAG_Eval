# from cohere_ret.cohere_ret import cohere_retriever
# from cohere_ret.generator import cohere_generator
# from gemini.retrieve import gemini_retriever
# from openai_class.retriever import openai_retriever
# from openai_class.generator import openai_generator
# from voyageai_ret.retrieve import voyage_retriever
# from gemini.generator import gemini_generator
# from utils.prepare_ground_truth import LatestGroundTruthCSV
# from evaluator.ret_eval import rag_app, retriever_evaluator
# from utils.chunk_scorer import score_chunk
from utils.prepare_expected_chunks import prepare_expected_chunks
# from evaluator.evaluator import RAG_eval



# query = "How do I register for controlled or semi-controlled drugs custody?"
# ret = voyage_retriever()

# chunks = ret.get_Chunks(query)
# print(chunks)

# gen = gemini_generator()
# gen = cohere_generator()
# response = gen.generate(query, chunks)
# print(response)


# csv_filepath = 'GroundTruths_Dataset - Sheet1.csv'
# json_filepath = 'url_chunk_mapping_1000.json'

# processor = LatestGroundTruthCSV(csv_filepath, json_filepath)
# ground_truth = processor.get_latest_ground_truth()

prepare_expected_chunks('data_1000_v2.0.json', 'data_1000_scored.json')

# for i in range(len(ground_truth["query"])):
#     ground_truth["expected_chunks"][i] = [{"text":expected_chunk, "title":expected_chunk, "expected_score":score_chunk(expected_chunk,ground_truth["expected_response"][i])} for expected_chunk in ground_truth["expected_chunks"][i] ]

# import json

# def write_dict_to_json_file(dictionary):
#     """
#     Writes a dictionary to a JSON file named 'data.json'.

#     Args:
#         dictionary (dict): The dictionary to be written to the JSON file.
#     """
#     try:
#         with open("data_1000.json", "w") as json_file:
#             json.dump(dictionary, json_file, indent=4)  # Pretty-print JSON with 4-space indent
#         print("Dictionary successfully written to data.json")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Example usage
# write_dict_to_json_file(ground_truth)



# rag_app = rag_app(ret, gen,ground_truth["expected_response"],ground_truth["query"])

# res = rag_app.retrieve_and_generate("How do I register for controlled or semi-controlled drugs custody?", k=10)
# print(res)
# print(type(res))
# ret_eval = retriever_evaluator(name="eval_gemini_cohere",ground_truth=ground_truth,rag_app=rag_app)

# # ret_eval.run()
# ret_eval.leaderboard()
# print(ret_eval.leaderboard())
