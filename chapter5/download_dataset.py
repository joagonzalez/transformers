from datasets import load_dataset
from pprint import pprint


dataset = load_dataset('glue', 'cola', cache_dir='data_test')
pprint(dataset)

