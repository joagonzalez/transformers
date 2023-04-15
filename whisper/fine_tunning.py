import os
import librosa.display
from pprint import pprint
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor


# Dataset 

common_voice = DatasetDict()

# https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/es/train
language = 'es'
dataset_name = "mozilla-foundation/common_voice_11_0"

common_voice["train"] = load_dataset(dataset_name, language, split="train+validation", use_auth_token=True, )
common_voice["test"] = load_dataset(dataset_name, language, split="test", use_auth_token=True)

pprint(common_voice)

# Feature Extractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# Tokenizer

tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", 
        language="Spanish", 
        task="transcribe"
    )

pprint(common_voice["train"][0])
input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Tokens:                {labels}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")


y = common_voice["train"][0]['audio']['array']
print(f'Signal vector: {y}')


# model was trained with 16khz audio files, then we need to adapt ours to the same format/sampling rate
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# re load first file
pprint(common_voice['train'][0])


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


print(f"Number of processes: {os.cpu_count()}")

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=os.cpu_count())