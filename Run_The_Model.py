import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TextDataset, \
    T5ForConditionalGeneration, \
    T5Tokenizer

import LossCallBack
from SkillsTitleDataset import SkillsTitleDataset

matplotlib.use('TkAgg')


"""
This class is dedicated for evaluating the model, in this class we load our trained mode,
and then we generate course titles from the test dataset, and we plot the skills, the original title and the generated 
title. All u have to do is to click RUN
"""

# Loading the model
model_path = 'seq2seq_final_project_t5'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Prepping the data
train_data_path = "courses_train_dataset_v2.csv"
train_data = pd.read_csv(train_data_path)

validation_data_pth = "courses_validation_dataset_v2.csv"
validation_data = pd.read_csv(validation_data_pth)

test_data_path = "courses_test_dataset_v2.csv"
test_data = pd.read_csv(test_data_path)

train_dataset = SkillsTitleDataset(train_data, tokenizer)
validation_dataset = SkillsTitleDataset(validation_data, tokenizer)
test_dataset = SkillsTitleDataset(test_data, tokenizer)

test_dataloader = DataLoader(test_dataset, batch_size=1)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./seq2seq_final_project_t5",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=1,  # Define the logging steps
    predict_with_generate=True,
    evaluation_strategy="steps",
    eval_steps=1,  # Evaluate every 2 steps
)

# Defining Callbacks
loss_callback = LossCallBack.LossCallback()
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    callbacks=[loss_callback, early_stopping]
)


def evaluate_model(model, tokenizer, eval_dataloader):
    # Set model to evaluation mode
    model.eval()
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    test_losses = []
    # Iterate over test dataset and generate predictions
    for batch in eval_dataloader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        # Generate predictions
        with torch.no_grad():
            output = model.generate(
                input_ids,
                num_beams=4,
            )

        # Decode predictions and labels
        decoded_preds = tokenizer.decode(output[0], skip_special_tokens=True)
        decoded_labels = tokenizer.decode(labels[0], skip_special_tokens=True)

        # Calculate loss
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        total_loss += loss.item()
        test_losses.append(loss.item())

        print("Skills:", input_text[len("[Skills] "): -len(" [Title]")])
        print("Prediction:", decoded_preds)
        print("Label:", decoded_labels)
        print("Loss:", loss.item())
        print()

    # Calculate average loss
    average_loss = total_loss / len(eval_dataloader)
    print("Average Loss:", average_loss)

    # Plot the test losses
    plt.plot(test_losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.show()


# Call the function to evaluate the model
evaluate_model(model, tokenizer, test_dataloader)
