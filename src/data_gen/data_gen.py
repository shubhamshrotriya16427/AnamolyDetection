import json
import os
import multiprocessing
import concurrent.futures

import pandas as pd

from graph_feat.graph_feat import parse_graph_features
from stat_feat.stat_feat import parse_stat_feat

BATCH_SIZE = 1
MAX_THREADS = multiprocessing.cpu_count()


def parse_features(json):
    graph_features = parse_graph_features(json)
    statistical_features = parse_stat_feat(json)

    return graph_features | statistical_features


def file_parser(file):
    features = None
    with open(file, 'r') as my_file:
            json_data = json.load(my_file)
            features = parse_features(json_data)
            features["file_name"] = file.split("/")[-1]

    return features


def generate_dataset(parsed_data_directory, destination):
    if parsed_data_directory[-1] != "/":
        parsed_data_directory += "/"

    files = os.listdir(parsed_data_directory)
    files = sorted(files, key=lambda x: os.path.getsize(os.path.join(parsed_data_directory, x)))

    current_batch_idx = 0
    while current_batch_idx < len(files):
        data = []
        print(f"Processed - {current_batch_idx}")
        _files = files[current_batch_idx: current_batch_idx+BATCH_SIZE]
        current_batch_idx += BATCH_SIZE

        _destination = destination[:-4] + str(current_batch_idx) + destination[-4:]

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            outputs = [executor.submit(file_parser, parsed_data_directory + file) for file in _files]

            for output in outputs:
                try:
                    data.append(output.result())
                except Exception as e:
                    print(f"Error processing data {data}: {e}")

        print(f"Saving Data to {_destination}")
        df = pd.DataFrame(data)
        df.to_csv(_destination)
    
    
