# Foundational Model Tuning
This project is an exercise in fine-tuning a foundational LLM model. The primary goal of the project is to load a pretrained model from Hugging Face, evaluate how the model performs before training in a classification task, perform Parameter-Efficient Fine-Tuning (PEFT) on the model using the chosen dataset, and then evaluate the PEFT model's performance.

## Chosen Dataset
I chose the [AuthorMix](https://huggingface.co/datasets/hallisky/AuthorMix) dataset. This dataset consists of several "styles" (various authors), "categories" (the type of writing, e.g., blog, speech, etc.), and of course the sample text.

## Chosen Problem
The goal is to accurately label text according to a particular style (author) with the highest possible accuracy.

## Approach
To "solve" the stated problem, my general approach is to use the GPT-2 model (primarily because I can manipulate this model locally) along with the Hugging Face [AutoModelForSequenceClassification](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification) class to orient the GPT-2 LLM for classification. The training process will involve using LoRA (Low-Rank Adaptation) to tune the LLM against the AuthorMix training split and then evaluate it against the test split.

## Results
The results of the project can be found in `project.ipynb`. Generally, the best results I was able to achieve were by adjusting the LoRA configuration parameter "lora_alpha" to a value of 128. This particular configuration is a scaling factor that determines how much influence the adaptation has on the model's weights. The scaling factor of 128 was either the correct value or slightly too high (which can lead to overfitting), as in the last epoch my validation loss started to increase. This was an exercise in understanding the general shape of a solution, but I surmise that with more compute resources, more accurate classification may be achieved using larger, more complex models.