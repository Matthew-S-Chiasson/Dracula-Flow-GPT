from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
'''
import os
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["TORCH_HOME"] = "D:/torch_cache"
'''


# Load dataset from a local text file
dataset = load_dataset("text", data_files={"train": "GenaratedRaw.txt"}, split="train")

# Split the dataset into training and evaluation sets (90% train, 10% eval)
train_size = 0.9
train_test_split = dataset.train_test_split(test_size=1-train_size, seed=42)

# Get the train and eval datasets
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Load GPT-2 model and tokenizer from the Hugging Face model hub
model_name = "openai-community/gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the pad_token to be the same as eos_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenizing function
def tokenize_function(examples):
    # Tokenize and set labels to be the same as inputs (shifted)
    encodings = tokenizer(examples["text"], padding="max_length", truncation=True)
    encodings["labels"] = encodings["input_ids"]  # Set labels equal to input_ids for causal language modeling
    return encodings

# Apply tokenization to the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Output directory for saving checkpoints and logs
    overwrite_output_dir=True,
    num_train_epochs=5,  # Number of epochs
    per_device_train_batch_size=4,  # Adjust based on available memory
    gradient_accumulation_steps=1,  # Accumulate gradients for larger effective batch size
    fp16=False,  # Disable mixed precision for GTX 1080 Ti (adjust based on your GPU)
    evaluation_strategy="steps",
    eval_steps=50,  # Evaluate every 50 steps
    save_steps=500,  # Save model checkpoints every 500 steps
    save_total_limit=2,  # Keep only the last 2 model checkpoints
    logging_dir="./logs",  # Directory for logs
    logging_steps=10,  # Log every 10 steps
    logging_first_step=True,  # Log the first step
    learning_rate=5e-5,  # Learning rate
    weight_decay=0.01,  # Weight decay for regularization
    warmup_steps=500,  # Number of warmup steps
    report_to="none",  # Disable reporting to external platforms like TensorBoard
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use the tokenized training dataset
    eval_dataset=eval_dataset,    # Use the tokenized evaluation dataset
)

# Start the training process
trainer.train()

# Save the model after training
model.save_pretrained("./DraculaFlowGPT_1.0")
tokenizer.save_pretrained("./DraculaFlowGPT_1.0")
