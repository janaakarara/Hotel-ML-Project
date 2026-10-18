#hy download h2.csv yet2aked mn en 3adad el rows w cloumns
import os
import urllib.request
import pandas as pd
from config import H2_URL, DATA_FILE, DATA_DIR

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print(f"Downloading H2.csv from {H2_URL} ...")
        urllib.request.urlretrieve(H2_URL, DATA_FILE)
        print("Download finished successfully.")
    else:
        print("H2.csv already exists locally.")

def load_data() -> pd.DataFrame:
    download_data()
    df = pd.read_csv(DATA_FILE)
    
    # Assertions zay shoroot el project
    assert df.shape == (79330, 31), f"Expected shape (79330, 31), but got {df.shape}"
    print(f"Dataset loaded and verified: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

if __name__ == "__main__":
    df = load_data()
    print("Columns in H2 dataset:")
    print(df.columns.tolist())