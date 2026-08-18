import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def process_data(file_path, mode, max_len, PAD_TOKEN=0):
    """
    Process parquet data based on mode ('train' or 'evaluation').

    Args:
        file_path (str): Path to the parquet file.
        mode (str): Mode of operation ('train' or 'evaluation').
        max_len (int): Maximum length for padding or truncation.

    Returns:
        list: Processed data.
    """
    data = pd.read_parquet(file_path)
    data['sequence'] = data['history'].apply(lambda x: list(x)) + data['target'].apply(lambda x: [x])
    
    if mode == 'train':
        # Issue #166 v2 (no windowed): 与 Euclidean_Base 系列对齐 — 每用户只保留最后一个窗口
        # (完整 history + train.target), 不用前缀窗口采样 (windowed 已删除, n_train = n_users)
        process_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            process_data.append({
                'history': sequence[:-1],
                'target': sequence[-1]
            })
    elif mode == 'evaluation':
        process_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            process_data.append({
                'history': sequence[:-1],
                'target': sequence[-1]
            })
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")
    
    for item in process_data:
        item['history'] = pad_or_truncate(item['history'], max_len)

    return process_data

def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """
    Pad or truncate a sequence to a specified maximum length.

    Args:
        sequence (list): Input sequence.
        max_len (int): Maximum length for padding or truncation.
        PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.

    Returns:
        list: Padded or truncated sequence.
    """
    if len(sequence) > max_len:
        # Truncate sequence
        return sequence[-max_len:]
    else:
        # Left pad sequence with PAD_TOKEN
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence
    
def item2code(code_path, codebook_size):
    data = np.load(code_path, allow_pickle=True)
    item_to_code = {}
    code_to_item = {}

    for index, code in enumerate(data):
        offsets = [c + sum(codebook_size[0:i]) + 1 for i,c in enumerate(code)]
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1
    return item_to_code, code_to_item

class GenRecDataset(Dataset):
    def __init__(self, dataset_path, code_path, mode, codebook_size, max_len, PAD_TOKEN=0):
        """
        Initialize the GenRecDataset.
        Args:
            dataset_path (str): Path to the dataset file.
            code_path (str): Path to the item-to-code mapping file.
            mode (str): Mode of operation ('train' or 'evaluation').
            max_len (int): Maximum length for padding or truncation.
            PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.
        """
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.mode = mode
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.codebook_size = codebook_size
        # Load item-to-code mapping
        self.item_to_code, self.code_to_item = item2code(code_path, self.codebook_size)
        # Process the dataset
        self.data = self._prepare_data()
        

    def _prepare_data(self):
        """
        Process the dataset and convert items to codes.
        Returns:
            list: Processed data with items converted to codes.
        """
        processed_data = process_data(
            self.dataset_path, self.mode, self.max_len, self.PAD_TOKEN
        )

        for item in processed_data:
            item['history'] = [self.item_to_code.get(x, np.array([self.PAD_TOKEN]*4)) for x in item['history']]
            item['target'] = self.item_to_code.get(item['target'], np.array([self.PAD_TOKEN]*4))
        return processed_data
    
    def __getitem__(self, index):
        """
        Get a single data item by index.
        Args:
            index (int): Index of the data item.
        Returns:
            dict: A dictionary containing 'history' and 'target'.
        """
        return self.data[index]
    
    def __len__(self):
        """
        Get the total number of data.
        Returns:
            int: Total number of data.
        """
        return len(self.data)
    
    
        