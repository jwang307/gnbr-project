import glob
import multiprocessing
import os
from itertools import chain
from pathlib import Path

import tqdm
from datasets
from huggingface_hub import HfApi
from transformers import AutoTokenizer

user_id = HfApi().whoami()["name"]

data_dir = "./abstracts_v2"

# Get a list of all the txt files in the directory
file_paths = [str(file_path) for file_path in Path(data_dir).glob("*.txt")]

# Define a function to read each txt file and return its contents
def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Use the `datasets.Dataset.from_generator()` method to load the txt files into a Hugging Face dataset
raw_datasets = datasets.Dataset.from_generator(
    generator=lambda: (read_txt_file(file_path) for file_path in tqdm.tqdm(file_paths)),
    output_types={"text": datasets.Value("string")},
    # Set the number of elements in the dataset to the number of txt files in the directory
    # to avoid unnecessary memory allocation
    num_rows=len(file_paths),
)

# tokenizer = AutoTokenizer.from_pretrained(f"{user_id}/{tokenizer_id}")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
num_proc = multiprocessing.cpu_count()
print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

def group_texts(examples):
    tokenized_inputs = tokenizer(
       examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

# preprocess dataset
tokenized_datasets = raw_datasets.map(group_texts, batched=True, remove_columns=["text"], num_proc=num_proc)

def group_texts_2(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

tokenized_datasets = tokenized_datasets.map(group_texts_2, batched=True, num_proc=num_proc)
# shuffle dataset
tokenized_datasets = tokenized_datasets.shuffle(seed=34)

print(f"the dataset contains in total {len(tokenized_datasets)*tokenizer.model_max_length} tokens")

dataset_id=f"{user_id}/processed_bert_dataset"
tokenized_datasets.push_to_hub(f"{user_id}/processed_bio_bert_tiny_dataset")
