from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

tokenizer = AutoTokenizer.from_pretrained("tokenizer")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Define the dataset
dataset = load_dataset("yashpatil/processed_bio_bert_tiny_dataset")

# Define the BERT Tiny model
model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")

model.resize_token_embeddings(len(tokenizer))

# Define the training arguments
training_args = TrainingArguments(output_dir="./biobert_tiny_results")

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Start the training process
trainer.train()

model.push_to_hub('biobert-tiny-model')
