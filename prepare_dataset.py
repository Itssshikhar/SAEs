import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

def process(example):
    text = example['text']  # Access the 'text' field from the dataset example
    ids = enc.encode_ordinary(text)  # Encode the text to tokens
    ids.append(enc.eot_token)  # Append the end-of-text token
    return {'ids': ids, 'len': len(ids)}

if __name__ == '__main__':
    # Load enwik8 dataset (assuming you have it in a suitable format)
    dataset = load_dataset("enwik8")  # Load your dataset

    # Concatenate all text from the dataset
    full_text = ''.join(dataset['train']['text'])

    # Split the full text into train, validation, and test
    train_text = full_text[:90_000_000]
    val_text = full_text[90_000_000:95_000_000]
    test_text = full_text[95_000_000:]

    # Create corresponding datasets directly from text
    train_dataset = Dataset.from_dict({'text': [train_text]})
    val_dataset = Dataset.from_dict({'text': [val_text]})
    test_dataset = Dataset.from_dict({'text': [test_text]})

    # Tokenize each split
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
    }

    tokenized = {}
    for split_name, split_data in datasets.items():
        tokenized[split_name] = split_data.map(
            process,
            remove_columns=['text'],
            desc=f'tokenizing {split_name}',
            num_proc=4,
        )
    # Concatenate and save the tokenized data
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # Using 16-bit integers to store tokens
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        idx = 0
        for batch_idx in tqdm(range(1), desc=f'writing {filename}'):
            batch = dset.with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
