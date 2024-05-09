# Course-Title-Generation-Project

Objective :
This project aims to utilize Deep Learning and LLM techniques and algorithms, to 
solve the problem of generating courses titles from courses skills, by automating the 
process.
The objective of this project is to fine-tune a T5 (Text-To-Text Transfer Transformer) 
model to generate course titles based on given course skills. The model aims to learn 
the relationship between course skills and their corresponding titles and generate 
accurate and relevant titles for unseen course skills 


Data :
We use this dataset: https://www.kaggle.com/datasets/azraimohamad/coursera-course-data
The dataset file contains Title, Organization, Skills, Ratings, Review and Metadata. 
The dataset used for training and evaluation consists of course skills along with their 
corresponding titles. In addition, its split into a training set and a test set, with 80% of 
the data used for training and 20% for testing, and we split the test data into test data 
and validation data, with 50% of the data for each one (test 10% and validation 10%)
using the "train_test_split" function from the scikit-learn library, and we save the data 
on separate files(training, validation and test). Each data point contains information 
about the skills required for the course and its title .
We defines a custom dataset class called "SkillsTitleDataset" that inherits from the 
"torch.utils.data.Dataset" class. This class is used to train a sequence-to-sequence 
model for generating course titles from course content. Its structured format includes 
two distinct segments, with the course skills segment enclosed within [Skills] markers 
and the course title presented without tags.


File Descriptions and Functionality 
- Project_seq2seq.py: This file contains the code to train the model and evaluate 
it. It defines the trainer, trainer arguments, tokenizer, and other things that 
related to the training and evaluation process.
- SkillsTitleDataset.py: This file defines a specific format for the data used in 
training the model. It prepares the data in a structured manner, ensuring 
compatibility with the model's input requirements.
- DataPreProcessing.py: This file handles data preprocessing tasks by splitting 
the data into training, validation, and test sets. It then saves each set into a 
separate CSV file for later use in training and evaluation.
- LossCallBack.py: The losscallback.py file contains code to append the train 
loss and validation loss to lists during the training process. It serves as a 
callback function to track the model's performance metrics over epochs.
- Run_The_Model.py: This file provides a convenient way to load the pre-trained model provided with the files. It allows users to evaluate the model and 
observe the generation results without needing to modify or write additional 
code. Just run it!!
On this class we use the pre-trained model that called "seq2seq_final_project_t5". This model is on drive and here its link:
https://drive.google.com/file/d/1L8Xe4bCIpX-I9VeJ0qGJSTbjAUUKc90j/view
