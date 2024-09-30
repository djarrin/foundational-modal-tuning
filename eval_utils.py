import time

def evaluate_model(dataset, model, tokenizer, label2id, n=10):
    correct_predictions = 0

    for i in range(n):
        text = dataset['test'][i]['text']
        true_label = dataset['test'][i]['style']  # Assuming 'style' is the label field

        if isinstance(text, list):
            text = " ".join(text)

        predicted_class = get_prediction(model, text, tokenizer)

        if predicted_class.item() == label2id[true_label]:  # Convert tensor to scalar with .item()
            correct_predictions += 1

    accuracy = correct_predictions / n
    return accuracy

def get_prediction(model, text, tokenizer):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    outputs = model(**inputs)

    # Get the predicted class (assuming this is for classification)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1)

    return predicted_class