from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
print(f'dataset features: {raw_datasets["train"].features}')
print(raw_datasets['train'][:6])