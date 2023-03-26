from datasets import load_dataset
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer

# load dataset
raw_datasets = load_dataset("glue", "mrpc")


# select checkpoint for model and also tokenizer
checkpoint = "bert-base-uncased"

# instantiate pre trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# tokenize dataset to it can be used with the model as input
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# use traier module within transformer library
training_args = TrainingArguments("test-trainer")
print(f'Training arguments: {training_args}')

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# execute predictions using trained model on validation dataset in order to compute performance metrics
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

# compute performance metrics
import evaluate
import numpy as np

metric = evaluate.load("glue", "mrpc")

preds = np.argmax(predictions.predictions, axis=-1)
metric.compute(predictions=preds, references=predictions.label_ids)

print(metric)

# How to automate performance metrics computation
# def compute_metrics(eval_preds):
#     metric = evaluate.load("glue", "mrpc")
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# trainer = Trainer(
#     model,
#     training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )