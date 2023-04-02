import numpy as np
from pprint import pprint
from datasets import load_dataset, load_metric
from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer, 
        TrainingArguments,
        Trainer
    )



def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True)
  
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



if __name__ == '__main__':
    
    metric = load_metric('glue', 'cola')
    raw_datasets = load_dataset('glue', 'cola')
    pprint(raw_datasets)

    model_checkpoint = 'bert-base-cased'

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_dataset = raw_datasets.map(preprocess_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

    args = TrainingArguments(
        'joagonzalez/bert-fine-tuned-cola',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        num_train_epochs=4,
        weight_decay=0.1,
        push_to_hub=True,
        # push_to_hub_organization='joagonzalez'
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    trainer.train()
    
    trainer.push_to_hub('end of trainig')