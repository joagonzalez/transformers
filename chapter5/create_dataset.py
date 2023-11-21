import time
import math
import requests
import pandas as pd
from pathlib import Path
from pprint import pprint
from tqdm.notebook import tqdm
from datasets import load_dataset


GITHUB_TOKEN = "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"

headers = {"Authorization": f"token {GITHUB_TOKEN}"}

def fetch_issues(
    owner="huggingface",
    repo="datasets",
    num_issues=1_000,
    rate_limit=5_000,
    issues_path=Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        # Query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # Flush batch for next time period
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(
        f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl"
    )
    
fetch_issues()

issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")
pprint(issues_dataset)

sample = issues_dataset.shuffle(seed=666).select(range(3))
pprint(sample)

# Print out the URL and pull request entries
for url, pr, number in zip(sample["html_url"], sample["pull_request"], sample["number"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n")
    print(f">> Number: {number}")
  
# add new column  
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)

pprint(issues_dataset)

# dataset augmentation adding comments to issues
def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]

# Test our function works as expected
# Depending on your internet connection, this can take a few minutes...
issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)

