import os
from datetime import datetime
import pandas as pd

DATA_FOLDER = r"/Users/adityamagarde/Documents/UCR/CS240_NR/AnamolyDetection/data/csvs/rcc_05/"
OUTPUT_PATH = r"/Users/adityamagarde/Documents/UCR/CS240_NR/AnamolyDetection/data/csvs/m5.csv"

if __name__ == "__main__":
    csv_files = os.listdir(DATA_FOLDER)
    
    dfs = []
    for csv_file in csv_files:
        if csv_file[-3:] == "csv":
            _df = pd.read_csv(DATA_FOLDER + csv_file)
            dfs.append(_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop(["Unnamed: 0"], axis=1)

    combined_df['date'] = combined_df['file_name'].apply(lambda x: datetime.strptime(x.split('.')[1], "%Y%m%d").strftime("%Y-%m-%d"))
    combined_df['time'] = combined_df['file_name'].apply(lambda x: datetime.strptime(x.split('.')[2], "%H%M").strftime("%H:%M"))

    combined_df_sorted = combined_df.sort_values(by='file_name', ascending=True)
    combined_df_sorted.reset_index(drop=True, inplace=True)


    combined_df_sorted.to_csv(OUTPUT_PATH)


2006