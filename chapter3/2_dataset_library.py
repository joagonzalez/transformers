from datasets import load_dataset
from transformers import AutoTokenizer


raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
print(f'dataset features: {raw_datasets["train"].features}')
print(raw_datasets['train'][14])
print(raw_datasets['train'][86])

# if we want to comapre two sentences, tokenizer api is ready to deal with that problem
print(f"\nTrain data id 15: {raw_datasets['train'][14]}")
print(f"\nSentence 1: {raw_datasets['train'][14]['sentence1']}")
print(f"\nSentence 2: {raw_datasets['train'][14]['sentence2']}")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokens = tokenizer(raw_datasets['train'][14]['sentence1'], raw_datasets['train'][14]['sentence2'])
print(f'Tokens: {tokens}')
print(f'Decoding tokens: {tokenizer.convert_ids_to_tokens(tokens["input_ids"])}')