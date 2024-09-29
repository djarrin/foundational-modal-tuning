import transformers
import torch
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding


def get_lora_target_modules(model):
    target_modules = set()
    for name, module in model.named_modules():
        # Check if the module is of a supported type
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, transformers.pytorch_utils.Conv1D)):
            # Exclude containers like ModuleDict
            if not isinstance(module, torch.nn.ModuleDict):
                # Check for common attention layer names in the module name
                if any(key in name.lower() for key in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'c_attn', 'c_proj', 'fc1', 'fc2', 'wte']):
                    target_modules.add(name)
    return list(target_modules)

def generate_training_args(model_save_name):
    return TrainingArguments(
    output_dir=f"./results/{model_save_name}",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Ensure logits and labels are on CPU and converted to numpy arrays
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# Create validation split function
def create_validation_split(dataset_dict):
    # Check if 'train' split exists
    if 'train' in dataset_dict:
        # Perform train_test_split on the 'train' dataset
        split_dataset = dataset_dict['train'].train_test_split(test_size=0.1)
        # Update the dataset dictionary
        dataset_dict['train'] = split_dataset['train']
        dataset_dict['validation'] = split_dataset['test']
    else:
        raise ValueError("No 'train' split found in the dataset.")
    return dataset_dict