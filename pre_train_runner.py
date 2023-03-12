from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertForPreTraining,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    Trainer,
    TrainingArguments,
)

# Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

# Define the dataset
dataset = load_dataset("yashpatil/processed_bio_bert_tiny_dataset")

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Define the BERT Tiny model
model = BertForPreTraining.from_pretrained("prajjwal1/bert-tiny")

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./biobert_tiny_results",
    evaluation_strategy="steps",
    eval_steps=5000,
    save_steps=5000,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    warmup_steps=10000,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    load_best_model_at_end=True,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Start the training process
trainer.train()

model.push_to_hub('biobert-tiny-model')
