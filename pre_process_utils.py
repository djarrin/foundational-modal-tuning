def preprocess_function(examples, tokenizer, label2id, return_tensors=False):
    tokenized_inputs = tokenizer(
        examples["text"],
        padding="max_length",  # Ensure padding is applied
        truncation=True,
        max_length=512,
        return_tensors="pt" if return_tensors else None
    )
    # Ensure labels are integers
    if "style" in examples:
        tokenized_inputs["labels"] = [label2id[label] for label in examples["style"]]
        
    return tokenized_inputs

def tokenize_splits(dataset, model_tokenizer):
    tokenized_datasets = {}
    for split in dataset.keys():
        # Tokenize the split
        tokenized_split = dataset[split].map(model_tokenizer, batched=True)
        
        # Set the format of the dataset to PyTorch tensors
        tokenized_split.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        # Store the processed split
        tokenized_datasets[split] = tokenized_split

    return tokenized_datasets