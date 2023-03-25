from transformers import BertModel, AutoTokenizer

model = BertModel.from_pretrained("bert-base-cased")

sequences = [
                "Hello!", 
                "Cool.", 
                "Nice!",
                "Using a Transformer network is simple"
             ]

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model_squences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

print(f'Sequence: {sequences}')
print(f'Endoded Sequence: {model_squences["input_ids"]}')
print(f'Model output: {model(model_squences["input_ids"])}')


'''
Instead of using tokenizer pipeline, we can do the whole process manually:
'''
print('-------------')
sentence = sequences[3]
tokens = tokenizer.tokenize(sentence)
ids = tokenizer.convert_tokens_to_ids(tokens)
decode = tokenizer.decode(ids)

print(f'Original sentence: {sentence}')
print(f'Tokens using sub-word tokenizer approach: {tokens}')
# 101 and 102 are START/END special SYMBOLS: [  101,  7993,   170, 13809, 23763,  2443,  1110,  3014,   102]
print(f'From tokens to IDs: {ids}')
print(f'Decoded from IDS: {decode}')