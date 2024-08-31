import json
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM, Trainer, TrainingArguments

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your dataset
with open('processed_documents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Flatten the data into a single list of chunks
all_chunks = [chunk for doc in data for chunk in doc['chunks']]

# Create a Hugging Face Dataset
dataset = Dataset.from_dict({"text": all_chunks})

# Use a valid model ID
model_id = "meta-llama/Meta-Llama-3-8B"  # Ensure this model is accessible on Hugging Face

# Load the tokenizer
try:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)  # Exit if tokenizer fails to load

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

try:
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
except Exception as e:
    print(f"Error during tokenization: {e}")
    exit(1)

# Split into train and test datasets
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Load the model and move it to the correct device
try:
    model = LlamaForCausalLM.from_pretrained(model_id).to(device)
    print("Model loaded successfully.")
    print("Model device:", next(model.parameters()).device)  # Verify model on GPU
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# If tokenizer was modified, resize model embeddings
model.resize_token_embeddings(len(tokenizer))

# Set training arguments with memory optimizations
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Reduced batch size
    per_device_eval_batch_size=1,   # Reduced evaluation batch size
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Start fine-tuning
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# Save the fine-tuned model
model.save_pretrained("fine-tuned-llama3.1")
tokenizer.save_pretrained("fine-tuned-llama3.1")

print("Fine-tuning complete and model saved!")