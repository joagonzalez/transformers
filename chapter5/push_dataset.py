from pprint import pprint
from datasets import load_dataset
from huggingface_hub import notebook_login


notebook_login()

issues_with_comments_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")
issues_with_comments_dataset.push_to_hub("github-issues")

remote_dataset = load_dataset("joagonzalez/github-issues", split="train")
pprint(remote_dataset)