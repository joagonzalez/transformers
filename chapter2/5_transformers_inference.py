from transformers import BertModel, AutoTokenizer

model = BertModel.from_pretrained("bert-base-cased")

sequences = ["Hello!", "Cool.", "Nice!"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model_squences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

print(f'Sequence: {sequences}')
print(f'Endoded Sequence: {model_squences["input_ids"]}')

print(f'Model output: {model(model_squences["input_ids"])}')