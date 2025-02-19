import os
import json
import argparse
import gzip
import shutil

from mrtparse import Reader

GZIP_EXT = ".gz"
JSON_EXT = ".json"


def unzip(mrt_zipped_path):
    unzipped_path = mrt_zipped_path[:-len(GZIP_EXT)]

    with gzip.open(mrt_zipped_path, 'rb') as mrt_zipped:
        with open(unzipped_path, 'wb') as mrt:
            shutil.copyfileobj(mrt_zipped, mrt)
    
    return unzipped_path


def convert_mrt_to_json(mrt_path, json_path):
    
    data = []

    for line in Reader(mrt_path):
        data.append(json.dumps([line.data], indent=2)[2:-2])
    
    with open(json_path, "w") as my_file:
        my_file.write("[\n" + ",\n".join(data) + "\n]\n")


def convert_file(mrt_path):
    # mrt_path = args.mrt_location
        
    if mrt_path[-3:] == GZIP_EXT:
        mrt_path = unzip(mrt_path)
        
    json_path = args.output_location
    if json_path is None:
        json_path = mrt_path + JSON_EXT
    
    convert_mrt_to_json(mrt_path, json_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mrt_location", help="Location of MRT file")
    parser.add_argument("-o", "--output_location", help="Location of JSON")
    parser.add_argument("-f", "--folder", help="Pass true if locations are of folders consisting files")

    args = parser.parse_args()

    if args.folder:
        #TODO(Adi): Write the code to read through folder and convert all files from mrt to json.
        files = os.listdir(args.mrt_location)

        for file in files:
            print(f"Converting files -- {file}")
            convert_file(args.mrt_location + file)
    else:
        convert_file(args.mrt_location)
        mrt_path = args.mrt_location

