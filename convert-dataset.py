import json
import os


def read_json_files_from_directory(directory_path):
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    json_data_list = []

    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            json_data_list.append(json_data)

    return json_data_list


dataSet = read_json_files_from_directory("./t2")
with open('3.5-turbo-dataset-t2.jsonl', 'w', encoding='utf-8') as jsonl_file:
    for entry in dataSet:
        json_line = json.dumps(entry, ensure_ascii=False)
        jsonl_file.write(json_line + '\n')
