import time

def evaluate_model(dataset, model, tokenizer, label2id, n=10):
    correct_predictions = 0

    # Iterate over the first 'n' samples of the dataset
    for i in range(n):
        # Get the input text and true label from the dataset
        text = dataset['test'][i]['text']
        true_label = dataset['test'][i]['style']  # Assuming 'style' is the label field

        # If the text is not a string, join it into a string
        if isinstance(text, list):
            text = " ".join(text)

        # Get the model's prediction
        predicted_class = get_prediction(model, text, tokenizer)

        # Check if the prediction is correct
        if predicted_class.item() == label2id[true_label]:  # Convert tensor to scalar with .item()
            correct_predictions += 1

    # Return the accuracy of the model on 'n' samples
    accuracy = correct_predictions / n
    return accuracy

def get_prediction(model, text, tokenizer):
    # Ensure the text is correctly passed as a string
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Perform the prediction (get the logits)
    outputs = model(**inputs)

    # Get the predicted class (assuming this is for classification)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1)

    return predicted_class