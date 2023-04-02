import torch
from pprint import pprint
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new, we add label for each sentence in sequences list
batch["labels"] = torch.tensor([1, 1]) # 1 means sentece is positive and 0 negative
pprint(batch)

optimizer = AdamW(model.parameters(), no_deprecation_warning=True)
loss = model(**batch).loss
loss.backward()
optimizer.step()