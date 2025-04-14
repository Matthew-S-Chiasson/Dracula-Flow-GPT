from transformers import GPT2Tokenizer
from datasets import load_dataset

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token to avoid warnings

# Load dataset
dataset_path = "GenaratedRaw.txt"  # Replace with your file path
dataset = load_dataset("text", data_files=dataset_path)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Save tokenized dataset
tokenized_dataset.save_to_disk("tokenized_DraculaFlow_dataset")
print("Tokenized dataset saved to disk!")
