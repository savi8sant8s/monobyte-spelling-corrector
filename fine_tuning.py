from transformers import (
    EarlyStoppingCallback,
    ByT5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    Adafactor
)
from datasets import load_dataset
import argparse

def get_prefix(dataset_name):
    if dataset_name == 'iam':
        return "To correct"
    elif dataset_name == "bressay":
        return "Corrija"
    elif dataset_name == "rimes":
        return "Corriger"
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

def get_max_length(dataset_name):
    if dataset_name == 'iam':
        return 150
    elif dataset_name == "bressay":
        return 200
    elif dataset_name == "rimes":
        return 150
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

def preprocess_function(examples, tokenizer, prefix, token_length):
    prediction = [f"{prefix}: {p}" for p in examples['prediction']]
    ground_truth = examples['ground_truth']
    model_inputs = tokenizer(
        prediction, 
        max_length=token_length + 10,
        truncation=True,
        padding='max_length'
    )
    labels = tokenizer(
        ground_truth, 
        max_length=token_length + 10,
        truncation=True,
        padding='max_length'
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main(model_name, dataset_name, output_folder):
    dataset = load_dataset('csv', data_files={ 'train': f'datasets/{dataset_name}/train_data.csv' })
    dataset_full = dataset['train'].train_test_split(shuffle=True, test_size=0.1)
    dataset_train, dataset_valid = dataset_full['train'], dataset_full['test']

    tokenizer = ByT5Tokenizer.from_pretrained(model_name)

    prefix = get_prefix(dataset_name)
    token_length = get_max_length(dataset_name)

    tokenized_train = dataset_train.map(
        lambda examples: preprocess_function(examples, tokenizer, prefix, token_length),  
        batched=True,
        num_proc=4
    )
    tokenized_valid = dataset_valid.map(
        lambda examples: preprocess_function(examples, tokenizer, prefix, token_length),  
        batched=True,
        num_proc=4
    )

    model = T5ForConditionalGeneration.from_pretrained(model_name, from_flax=True)

    out_dir = output_folder
    batch_size = 8
    epochs = 15

    training_args = TrainingArguments(
        output_dir=out_dir,               
        num_train_epochs=epochs,              
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,               
        weight_decay=0.01,               
        logging_dir=out_dir,            
        save_strategy='epoch',
        evaluation_strategy='epoch',
        metric_for_best_model='eval_loss',                
        save_total_limit=3,
        report_to='tensorboard',
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        learning_rate=1e-4,
    )

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=tokenized_train,       
        eval_dataset=tokenized_valid, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        optimizers=(Adafactor(model.parameters(), relative_step=False, lr=training_args.learning_rate), None)
    )

    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='monobyte/byt5-mono-pt-v1')
    parser.add_argument('--dataset', type=str, default='bressay')
    parser.add_argument('--output_folder', type=str, default='models/bressay')
    
    args = parser.parse_args()
    
    main(args.model, args.dataset, args.output_folder)
