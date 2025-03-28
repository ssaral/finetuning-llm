import os
import math
import argparse
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model.")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model identifier (e.g., 'gpt2').")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name (e.g., 'wikitext') or path to a custom dataset file (CSV/JSON).")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Configuration name for the dataset (if applicable).")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column (for custom datasets).")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length (block size) for training.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached tokenized datasets.")
    return parser.parse_args()

def load_custom_dataset(file_path, text_column):
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

def tokenize_and_group(examples, tokenizer, block_size):
    outputs = tokenizer(examples["text"], return_special_tokens_mask=False)
    concatenated = sum(outputs["input_ids"], [])
    total_length = len(concatenated)
    total_length = (total_length // block_size) * block_size
    result = {"input_ids": [concatenated[i: i + block_size] for i in range(0, total_length, block_size)]}
    return result

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    try:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    except Exception as e:
        print("Could not load the dataset from the Hugging Face hub; trying as a custom file...")
        dataset = load_custom_dataset(args.dataset_name, args.text_column)

    if "text" not in dataset["train"].column_names:
        for split in dataset.keys():
            dataset[split] = dataset[split].rename_column(args.text_column, "text")

    def tokenize_function(examples):
        return tokenize_and_group(examples, tokenizer, args.max_length)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=not args.overwrite_cache,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results_causal",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    
    if "eval_loss" in eval_results:
        perplexity = math.exp(eval_results["eval_loss"])
        print(f"Perplexity: {perplexity}")

    output_dir = "./fine_tuned_causal_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()
