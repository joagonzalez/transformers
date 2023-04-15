from pprint import pprint
from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

pprint(drug_dataset)

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
pprint(drug_sample)
pprint(drug_sample[:1])

################################################
print('------------')
# create validation set
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
pprint(drug_dataset_clean)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]

pprint(drug_dataset_clean)
