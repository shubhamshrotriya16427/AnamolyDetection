import os
import requests
import gzip
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import re
from mrtparse import Reader
import json
import pandas as pd
import glob

# Configuration
RRC = "04"
TYPE = "updates"  # "updates" or "bview"
DATES = []  # Example list of dates ["2003-01-23"] , Format - YYYY-MM-DD
output_folder = os.path.join("data", "raw", f"output_{RRC}")
# Base directory set to data/raw
base_dir = os.path.join("data", "raw", f"updatesDownload_{RRC}")
os.makedirs(base_dir, exist_ok=True)

# Function to list files at base_url
def list_files(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.text

        # Assuming the server lists files in a simple format or HTML
        # This regex might need to be adjusted based on actual page content
        pattern = re.compile(r'href="([^"]*.gz)"')
        return pattern.findall(content)
    except Exception as e:
        print(f"Error listing files at {url}: {e}")
        return []

# Function to download a file
def download_file(url, filename):
    local_path = os.path.join(base_dir, filename)

    # Skip download if file already exists
    if os.path.exists(local_path):
        print(f"File {filename} already exists, skipping.")
        return

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        extract_file(local_path)
        print(f"Downloaded and extracted {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def extract_file(path_to_gz):
    with gzip.open(path_to_gz, 'rb') as f_in:
        extracted_path = path_to_gz[:-3]  # Removes the .gz extension
        with open(extracted_path, 'wb') as f_out:
            f_out.write(f_in.read())
    os.remove(path_to_gz)  # Delete the original .gz file
    convert_to_json(extracted_path)  # Convert to JSON

def convert_to_json(path_to_file):
    file_name = os.path.basename(path_to_file)
    json_file_path = os.path.join(base_dir, file_name + '.json')
    with open(json_file_path, 'w') as outfile:
        outfile.write('[\n')
        i = 0
        for entry in Reader(path_to_file):
            if i != 0:
                outfile.write(',\n')
            outfile.write(json.dumps(entry.data, indent=2))
            i += 1
        outfile.write('\n]\n')
    
    print(f"Converted {path_to_file} to JSON: {json_file_path}")
    os.remove(path_to_file)  # Delete the original MRT file
    print(f"Deleted original MRT file: {path_to_file}")

def download_for_dates(dates):
    for date_str in dates:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        year = date_obj.strftime("%Y")
        month = date_obj.strftime("%m")
        base_url = f"https://data.ris.ripe.net/rrc{RRC}/{year}.{month}"
        files = list_files(base_url)
        start_date_str = date_obj.strftime("%Y%m%d")
        pattern = re.compile(rf"{TYPE}\.{start_date_str}\.\d{{4}}\.gz")
        matching_files = [filename for filename in files if pattern.match(filename)]
        
        # Download files in parallel for each date
        with ThreadPoolExecutor(max_workers=10) as executor:
            for filename in matching_files:
                url = f"{base_url}/{filename}"
                executor.submit(download_file, url, filename)

def get_dates_from_filenames(directory):
    # Define a regular expression pattern to extract the date from the filename
    filename_pattern = re.compile(r'updates\.(\d{4})(\d{2})(\d{2})\.\d{4}\.json')
    dates = set()

    for filename in os.listdir(directory):
        match = filename_pattern.match(filename)
        if match:
            # Extract date components from the filename
            year, month, day = match.groups()
            # Convert to datetime to ensure correctness and then format to desired string format
            date_str = datetime(year=int(year), month=int(month), day=int(day)).strftime('%Y-%m-%d')
            dates.add(date_str)
    
    # Convert set to a sorted list
    return sorted(list(dates))

def delete_files_for_dates(directory, dates_to_delete):
    # Define a regular expression pattern to extract the date from the filename
    filename_pattern = re.compile(r'updates\.(\d{4})(\d{2})(\d{2})\.\d{4}\.json')
    
    for filename in os.listdir(directory):
        match = filename_pattern.match(filename)
        if match:
            year, month, day = match.groups()
            date_str = f"{year}-{month}-{day}"
            if date_str in dates_to_delete:
                file_path = os.path.join(directory, filename)
                os.remove(file_path)
                print(f"Deleted file: {filename}")

def extract_file_names_from_csv(output_folder):
    file_names = []  # List to store all file names
    # Iterate over all csv files in the output folder
    for file in os.listdir(output_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(output_folder, file)
            try:
                # Read the csv file
                df = pd.read_csv(file_path)
                # Check if 'file_name' column exists
                if 'file_name' in df.columns:
                    # Extend the list with the values from the 'file_name' column
                    file_names.extend(df['file_name'].tolist())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Returning unique file names if there could be duplicates
    return list(set(file_names))

def delete_files_in_directory(directory, filenames_or_patterns):
    for pattern in filenames_or_patterns:
        # Construct the full path with pattern
        full_path_pattern = os.path.join(directory, pattern)
        # Use glob to find all files matching the pattern
        files_to_delete = glob.glob(full_path_pattern)
        
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e.strerror}")

# Main
if __name__ == "__main__":
    download_for_dates(DATES)
    # dates_in_directory = get_dates_from_filenames(base_dir)
    # delete_files_for_dates(base_dir, dates_in_directory)
    # all_file_names = extract_file_names_from_csv(output_folder)
    # delete_files_in_directory(base_dir, all_file_names)