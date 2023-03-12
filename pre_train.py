import multiprocessing

from transformers import AutoTokenizer

# load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(f"{user_id}/{tokenizer_id}")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
num_proc = multiprocessing.cpu_count()
print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

# def group_texts(examples):
#     tokenized_inputs = tokenizer(
#        examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
#     )
#     return tokenized_inputs

# # preprocess dataset
# tokenized_datasets = raw_datasets.map(group_texts, batched=True, remove_columns=["text"], num_proc=num_proc)
# tokenized_datasets.features


# -------------------------------

# import glob
# import os

# import tqdm

# # specify the directory containing the files
# directory = "./abstracts_v2"

# # create an empty list to store the contents of each file
# file_contents = []

# # use the glob module to find all files in the directory
# files = glob.glob(os.path.join(directory, "*"))

# # loop over each file and read its contents
# for file in tqdm.tqdm(files):
#     with open(file, "r") as f:
#         content = f.read()
#         file_contents.append(content)

# # print the list of file contents
# print(len(file_contents))
