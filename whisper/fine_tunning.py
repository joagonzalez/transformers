from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()


# https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/es/train
language = 'es'
dataset_name = "mozilla-foundation/common_voice_11_0"


common_voice["train"] = load_dataset(dataset_name, language, split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(dataset_name, language, split="test", use_auth_token=True)

print(common_voice)
