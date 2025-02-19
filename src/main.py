from data_gen.data_gen import generate_dataset

DATASET_FOLDER_PATH = "" # Example: "/Users/adityamagarde/Documents/UCR/CS240_NR/AnamolyDetection/data/test_files"
OUTPUT_PATH = "" # Example: "/Users/adityamagarde/Documents/UCR/CS240_NR/AnamolyDetection/data/test_files/out.csv"

if __name__ == "__main__":
    generate_dataset(DATASET_FOLDER_PATH, OUTPUT_PATH)