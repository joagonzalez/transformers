from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Load dataset
raw_datasets = load_dataset("glue", "mrpc")

print(raw_datasets)


# Tokenize using map function and extending original dataset dictionary
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

# Adding data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Without data collator
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
result = [len(x) for x in samples["input_ids"]]


# With data collator
batch = data_collator(samples)
results_collator = {k: v.shape for k, v in batch.items()}

print(f'Result without dynamic padding: {result}')
print(f'Result with dynamic padding: {results_collator}')