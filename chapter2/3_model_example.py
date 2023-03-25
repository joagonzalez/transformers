'''
Based on an example from: https://huggingface.co/course/chapter2/2?fw=pt
'''

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification

# data
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

# Select a BERT based encoder model
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# Calculate tokens and embeddings from input
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(f'Inputs after Tokenizer: {inputs}')

# Using tokens/embeddings as input for the model without the head.
# The model is represented by its embeddings layer and the subsequent layers. 
# The embeddings layer converts each input ID in the tokenized input into a vector that
# represents the associated token.
# The subsequent layers manipulate those vectors using the attention mechanism to produce 
# the final representation of the sentences.
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)

print(outputs.last_hidden_state.shape)
# torch.Size([2, 16, 768])


'''
For our example, we will need a model with a sequence classification head 
(to be able to classify the sentences as positive or negative). 
So, we won’t actually use the AutoModel class, but AutoModelForSequenceClassification:
'''
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

'''
Now if we look at the shape of our outputs, the dimensionality will be much lower: 
the model head takes as input the high-dimensional vectors we saw before, 
and outputs vectors containing two values (one per label):
'''
print(outputs.logits.shape)
print(outputs.logits)

'''
Our model predicted [-1.5607, 1.6123] for the first sentence and [ 4.1692, -3.3464] for the second one. 
Those are not probabilities but logits, the raw, unnormalized scores outputted by the last layer of the model.
To be converted to probabilities, they need to go through a SoftMax layer
'''
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)

