import pandas as pd
import numpy as np
import os

def load_and_preprocess():

    data_path = r"C:\Users\Richa\Downloads\GeneratedLabelledFlows\TrafficLabelling"

    files = os.listdir(data_path)

    dfs = []

    print("Loading dataset...")

    for file in files:

        if file.endswith(".csv"):

            path = os.path.join(data_path, file)

            print("Loading:", file)

            df = pd.read_csv(path, encoding="latin1", low_memory=False)

            dfs.append(df)

    dataset = pd.concat(dfs, ignore_index=True)

    print("Dataset shape before preprocessing:", dataset.shape)

    drop_cols = ['Flow ID',' Source IP',' Destination IP',' Timestamp']

    dataset.drop(columns=drop_cols, inplace=True, errors="ignore")

    dataset.replace([np.inf,-np.inf], np.nan, inplace=True)

    dataset.dropna(inplace=True)

    dataset[' Label'] = dataset[' Label'].apply(lambda x: 0 if x=="BENIGN" else 1)

    print("Dataset shape after preprocessing:", dataset.shape)

    print("\nClass distribution:")

    print(dataset[' Label'].value_counts())

    return dataset