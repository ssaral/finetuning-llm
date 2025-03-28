import os
import argparse
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model on a dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model identifier (e.g., 'bert-base-uncased').")
    parser.add_argument("--dataset_name", type=str, default="glue", help="Dataset name (e.g., 'glue') or path to a custom dataset file (CSV/JSON).")
    parser.add_argument("--task_name", type=str, default="mrpc", help="For GLUE datasets, specify the task name (e.g., 'mrpc').")
    parser.add_argument("--custom_split", action="store_true", help="If using a custom dataset, assume it has 'train' and 'validation' splits.")
    parser.add_argument("--text_column", type=str, default="sentence", help="Name of the text column (for custom datasets).")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column (for custom datasets).")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    return parser.parse_args()

def load_custom_dataset(file_path, text_column, label_column):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        dataset = load_dataset("csv", data_files={"train": file_path})
    elif ext == ".json":
        dataset = load_dataset("json", data_files={"train": file_path})
    else:
        raise ValueError("Unsupported file format. Only CSV and JSON are supported.")
    
    split_dataset = dataset["train"].train_test_split(test_size=0.2)
    dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })
    return dataset

def tokenize_function(examples, tokenizer, text_column, max_length):
    return tokenizer(examples[text_column], truncation=True, max_length=max_length)

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)  # Adjust num_labels as needed
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.dataset_name.lower() == "glue":
        dataset = load_dataset("glue", args.task_name)
        text_column = "sentence1" if "sentence1" in dataset["train"].column_names else dataset["train"].column_names[0]
    else:
        # OR a file path to a custom dataset.
        dataset = load_custom_dataset(args.dataset_name, args.text_column, args.label_column)
        text_column = args.text_column

    # Tokenize it
    def tokenize_wrapper(examples):
        return tokenize_function(examples, tokenizer, text_column, args.max_length)

    tokenized_datasets = dataset.map(tokenize_wrapper, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",  # Set save strategy to "epoch" to match evaluation strategy
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # Save model and tokenizer in Hugging Face format
    output_dir = "./fine_tuned_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()
