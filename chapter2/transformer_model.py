'''
Every transformer NN hast two main components:
Config and Model. 
Config has details such number of layers required to build the model.
Model has details of the implementation for a specific network architecture. 
'''
from transformers import BertConfig, BertModel

# Building the config. Using default config it initializes weights with random values
# we can specify a checkpoint instead: model = BertModel.from_pretrained("bert-base-cased")
config = BertConfig()

# details of the config and weights can be found at:
# ~/.cache/huggingface/hub/models--bert-base-cased/snapshots/5532cc56f74641d4bb33641f5c76a55d11f846e0

# Building the model from the config
model = BertModel(config)

print(config)

model.save_pretrained("model_config")

