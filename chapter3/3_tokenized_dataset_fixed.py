from datasets import load_dataset
from transformers import AutoTokenizer


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_datasets = load_dataset("glue", "mrpc")

print(raw_datasets)

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

print(tokenized_dataset[:3])

'''
This works well, but it has the disadvantage of returning a dictionary 
(with our keys, input_ids, attention_mask, and token_type_ids, and values that are lists of lists). 
It will also only work if you have enough RAM to store your whole dataset during the tokenization 
(whereas the datasets from the 🤗 Datasets library are Apache Arrow files stored on the disk, 
so you only keep the samples you ask for loaded in memory).
'''

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)