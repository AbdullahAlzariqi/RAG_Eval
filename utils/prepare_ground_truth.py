import csv
import json
class LatestGroundTruthCSV:
    """
    A class to process and return the latest ground truth CSV with an added 'expected_chunks' column.
    """

    def __init__(self, csv_filepath, json_filepath):
        """
        Initializes the class with file paths for CSV and JSON files.

        Args:
            csv_filepath: The path to the ground truth CSV file.
            json_filepath: The path to the URL-chunk map JSON file.
        """
        self.csv_filepath = csv_filepath
        self.json_filepath = json_filepath
        self.ground_truth = None

    def read_csv_to_columns_with_keys(self, filepath):
        """
        Reads a CSV file and returns a dictionary where keys are from the first row 
        and values are lists representing columns (excluding the first row).
        """
        try:
            columns = {}
            with open(filepath, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                keys = next(reader, None)
                if keys is None:
                    return {}
                for key in keys:
                    columns[key] = []
                for row in reader:
                    for i, value in enumerate(row):
                        columns[keys[i]].append(value)
            return columns
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def read_json_to_dict(self, filepath):
        """
        Reads a JSON file and returns its contents as a Python dictionary.
        """
        try:
            with open(filepath, 'r',encoding='utf-8') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            print(f"Error: File not found at '{filepath}'")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None

    def add_expected_chunks(self, service_urls, url_chunk_map):
        """
        Adds expected chunks to a list based on service URLs and a URL-chunk map.
        """
        expected_chunks = []
        for i,url in enumerate(service_urls):
            if len(url.split(",")) == 1:
                chunks_list = url_chunk_map[url]
                expected_chunks.append(chunks_list)
            else:
                for u in url.split(","):
                    if len(expected_chunks) < i+1:
                        chunks_list = url_chunk_map[u.strip()]
                        expected_chunks.append(chunks_list)

                    chunks_list = url_chunk_map[u.strip()]
                    expected_chunks[i].extend(chunks_list)
        return expected_chunks

    def get_latest_ground_truth(self):
        """
        Processes the ground truth CSV and adds the 'expected_chunks' column, then returns the result.
        """
        if self.ground_truth is not None:
            return self.ground_truth  # Return cached result if available

        service_urls = self.read_csv_to_columns_with_keys(self.csv_filepath)["service_url"]
        if service_urls is None:
            return None  # Return None if CSV reading failed

        url_chunk_map = self.read_json_to_dict(self.json_filepath)
        if url_chunk_map is None:
            return None  # Return None if JSON reading failed

        expected_chunks_list = self.add_expected_chunks(service_urls, url_chunk_map)
        ground_truth = self.read_csv_to_columns_with_keys(self.csv_filepath)
        ground_truth["expected_chunks"] = expected_chunks_list

        self.ground_truth = ground_truth  # Cache the result
        return ground_truth

# # Example usage
# csv_filepath = 'GroundTruths_Dataset - Sheet1.csv'
# json_filepath = 'url_chunk_mapping_1000_v2.0.json'

# processor = LatestGroundTruthCSV(csv_filepath, json_filepath)
# latest_ground_truth = processor.get_latest_ground_truth()



# import json

# def write_dict_to_json(data, filename):
#     """
#     Write a dictionary to a JSON file.

#     :param data: Dictionary to write into a JSON file.
#     :param filename: Name of the JSON file to be created.
#     """
#     try:
#         with open(filename, 'w', encoding='utf-8') as json_file:
#             json.dump(data, json_file, ensure_ascii=False, indent=4)
#         print(f"Data successfully written to {filename}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Example usage

