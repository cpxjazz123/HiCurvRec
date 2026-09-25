import numpy as np
import pandas as pd
import json
import gzip
import os

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

dataset_name = "Instruments"
#=================================================================================#

data = open(f"./dataset/{dataset_name}/{dataset_name}.inter.json", 'r')

train_data = {}
val_data = {}
test_data = {}

for userID, item_sequence in json.load(data).items():
    item_sequence_ = []
    for item in item_sequence:
        item_sequence_.append(item+1)

    train_data[userID] = item_sequence_[:-2]
    val_data[userID] = item_sequence_[:-1]
    test_data[userID] = item_sequence_

# Print a sample of the split data
#print("training data:", list(train_data.items())[:5])
#print("validation data:", list(val_data.items())[:5])
#print("testing data:", list(test_data.items())[:5])

# Prepare data for train, validation, and test sets
def prepare_data(data_dict):
    rows = []
    for userID, item_sequence in data_dict.items():
        history = item_sequence[:-1]
        target = item_sequence[-1]
        rows.append({'user': userID, 'history': history, 'target': target})
    return pd.DataFrame(rows)

# Create dataframes for train, validation, and test sets
train_df = prepare_data(train_data)
#print("\nTraining data shape:", train_df.shape)
#print("the first 3 rows of training data:\n", train_df.head(3))
val_df = prepare_data(val_data)
#print("\nValidation data shape:", val_df.shape)
#print("the first 3 rows of validation data:\n", val_df.head(3))
test_df = prepare_data(test_data)
#rint("\nTesting data shape:", test_df.shape)
#print("the first 3 rows of testing data:\n", test_df.head(3))

# Save dataframes to parquet files
train_df.to_parquet(f'./dataset/{dataset_name}/train.parquet', index=False)
val_df.to_parquet(f'./dataset/{dataset_name}/valid.parquet', index=False)
test_df.to_parquet(f'./dataset/{dataset_name}/test.parquet', index=False)

#print("Data saved to parquet files.")

data.close()

#=================================================================================#
data = open(f"./dataset/{dataset_name}/{dataset_name}.item.json", 'r')
item_info = {}
for itemID, item_text in json.load(data).items():
    item_info[itemID] = item_text
#=================================================================================#
model = SentenceTransformer('sentence-transformers/sentence-t5-xl')
item_embeddings = []
for itemID, info in tqdm(item_info.items(), desc="Encoding items"):
    semantics = f"'title': {info['title']},\
                  'description': {info['description']},\
                  'brand': {info['brand']},\
                  'categories': {info['categories']}"
    #print(semantics)
    embedding = model.encode(semantics, show_progress_bar=False)
    item_embeddings.append({'ItemID': itemID, 'embedding': embedding.tolist()})

# Convert to DataFrame
item_emb_df = pd.DataFrame(item_embeddings)
print("\nItem embeddings DataFrame shape:", item_emb_df.shape)
print("The first 3 rows of item embeddings DataFrame:\n", item_emb_df.head(3))

# Save to parquet file
item_emb_df.to_parquet(f'./dataset/{dataset_name}/item_emb.parquet', index=False)

print("Item embeddings saved to item_emb.parquet.")
