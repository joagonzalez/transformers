from transformers import WhisperTokenizer
from datasets import load_dataset, DatasetDict

# Dataset 

common_voice = DatasetDict()

# https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/es/train
language = 'es'
dataset_name = "mozilla-foundation/common_voice_11_0"

common_voice["train"] = load_dataset(dataset_name, language, split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(dataset_name, language, split="test", use_auth_token=True)

print(common_voice)


# Tokenizer

tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", 
        language="Spanish", 
        task="transcribe"
    )

print(common_voice["train"][0])
input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")
