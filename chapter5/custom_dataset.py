from pprint import pprint
from datasets import load_dataset

# https://huggingface.co/course/chapter5/2?fw=pt
# load_dataset support loads datasets in csv|text|json|parquet formats

squad_it_dataset = load_dataset(
        "json", 
        data_files="SQuAD_it-train.json", 
        field="data"
    )

pprint(squad_it_dataset)
pprint(squad_it_dataset["train"][1])

# load both datasets at once

data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
pprint(squad_it_dataset)


# load both datasets at once using decompression feature from datasets library

data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
pprint(squad_it_dataset)