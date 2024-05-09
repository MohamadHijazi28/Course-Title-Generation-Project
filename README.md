# Course-Title-Generation-Project

File Descriptions and Functionality
• Project_seq2seq.py: This file contains the code to train the model and evaluate 
it. It defines the trainer, trainer arguments, tokenizer, and other things that 
related to the training and evaluation process.
• SkillsTitleDataset.py: This file defines a specific format for the data used in 
training the model. It prepares the data in a structured manner, ensuring 
compatibility with the model's input requirements.
• DataPreProcessing.py: This file handles data preprocessing tasks by splitting 
the data into training, validation, and test sets. It then saves each set into a 
separate CSV file for later use in training and evaluation.
• LossCallBack.py: The losscallback.py file contains code to append the train 
loss and validation loss to lists during the training process. It serves as a 
callback function to track the model's performance metrics over epochs.
• Run_The_Model.py: This file provides a convenient way to load the pretrained model provided with the files. It allows users to evaluate the model and 
observe the generation results without needing to modify or write additional 
code. Just run it!!
