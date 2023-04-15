from pprint import pprint
from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

pprint(drug_dataset)

# slicing

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
pprint(drug_sample[:3])


# check problems with column unamed 0
for split in drug_dataset.keys():
    print('analyzing uniqueness of unamed 0 column')
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
    
# change name of column
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)

pprint(drug_dataset)

pprint(drug_dataset.keys())
# pprint(drug_dataset.unique('drugName'))

# apply lowercase to confition filtering None ones
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

def filter_nones(x):
    return x["condition"] is not None

drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

pprint(drug_dataset)


drug_dataset = drug_dataset.map(lowercase_condition)
# Check that lowercasing worked
pprint(drug_dataset["train"]["condition"][:3])


# create new column
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset = drug_dataset.map(compute_review_length)
# Inspect the first training example
pprint(drug_dataset["train"][0])
pprint(drug_dataset)