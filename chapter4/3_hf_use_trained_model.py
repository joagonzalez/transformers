from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification

# data
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
    "i love it!"
]

# Select a custom trained model
checkpoint = "joagonzalez/bert-fine-tuned-cola"

# Calculate tokens and embeddings from input
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model_squences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# Instatiate a model
model = AutoModel.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


print(f'Sequence: {sequences}')
print(f'Endoded Sequence: {model_squences["input_ids"]}')
print(f'Model output: {model(model_squences["input_ids"])}')


'''
Instead of using tokenizer pipeline, we can do the whole process manually:
'''
print('-------------')
sentence = sequences[2]
tokens = tokenizer.tokenize(sentence)
ids = tokenizer.convert_tokens_to_ids(tokens)
decode = tokenizer.decode(ids)

print(f'Original sentence: {sentence}')
print(f'Tokens using sub-word tokenizer approach: {tokens}')
# 101 and 102 are START/END special SYMBOLS: [  101,  7993,   170, 13809, 23763,  2443,  1110,  3014,   102]
print(f'From tokens to IDs: {ids}')
print(f'Decoded from IDS: {decode}')
print(f'Decoded from IDS produced by pipeline: {tokenizer.decode(model_squences["input_ids"][2])}')
