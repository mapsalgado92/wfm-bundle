import json
import os

# Get the directory of the current module (objects/fetcher.py)
module_dir = module_dir = os.path.dirname(os.path.abspath(__file__))


def fetch_from_json(file_name):
    file_path = os.path.join(module_dir, file_name)
    return json.load(open(f"{file_path}.json", "r"))
