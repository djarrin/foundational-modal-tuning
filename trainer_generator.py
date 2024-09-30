import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
from functools import partial

from pre_process_utils import preprocess_function, tokenize_splits
from training_utils import get_lora_target_modules, create_validation_split, generate_training_args, compute_metrics

def generate_trainer(ds, lora_config, save_path):
    train_df = pd.DataFrame(ds['train'])
    unique_styles = train_df['style'].unique()

    load_dotenv("env.txt")
    hf_token = os.getenv('HF_API_TOKEN')
    login(hf_token)

    num_labels = len(unique_styles)
    id2label = {i: style for i, style in enumerate(unique_styles)}
    label2id = {v: k for k, v in id2label.items()}

    gpt_model_key = "distilbert/distilbert-base-uncased"
    gpt_model = AutoModelForSequenceClassification.from_pretrained(
        gpt_model_key,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        token=True,
    )

    gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_key, token=True)
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    gpt_model.config.pad_token_id = gpt_tokenizer.pad_token_id

    gpt_target_modules = get_lora_target_modules(gpt_model)

    gpt_model = get_peft_model(gpt_model, lora_config)

    gpt_2_tokenize = partial(preprocess_function, tokenizer=gpt_tokenizer, label2id=label2id)

    gpt_2_tokenized_ds = tokenize_splits(ds, gpt_2_tokenize)

    gpt_data_collator = DataCollatorWithPadding(tokenizer=gpt_tokenizer)

    gpt_2_tokenized_ds = create_validation_split(gpt_2_tokenized_ds)
    return Trainer(
        model=gpt_model,
        args=generate_training_args(save_path),
        train_dataset=gpt_2_tokenized_ds["train"],
        eval_dataset=gpt_2_tokenized_ds["test"],
        compute_metrics=compute_metrics,
        data_collator=gpt_data_collator,
    )
