from huggingface_hub import HfApi

user_id = HfApi().whoami()["name"]

print(f"user id '{user_id}' will be used during the example")



# from tqdm import tqdm
# from transformers import BertTokenizerFast

# # repositor id for saving the tokenizer
# tokenizer_id="bert-base-uncased-2022-habana"

# # create a python generator to dynamically load the data
# def batch_iterator(batch_size=10000):
#     for i in tqdm(range(0, len(raw_datasets), batch_size)):
#         yield raw_datasets[i : i + batch_size]["text"]

# # create a tokenizer from existing one to re-use special tokens
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
# bert_tokenizer.save_pretrained("tokenizer")
