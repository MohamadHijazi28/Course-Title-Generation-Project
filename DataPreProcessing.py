import pandas as pd
import random
from sklearn.model_selection import train_test_split


def get_random_permutation(skills_string):
    # Split the string into a list of skills
    skills_list = skills_string.split(", ")

    # Shuffle the skills list
    random.shuffle(skills_list)

    # Join the shuffled skills back into a string
    shuffled_skills_string = ", ".join(skills_list)

    return shuffled_skills_string


# Read the original dataset
data = pd.read_csv("courses_dataset_v2.csv")

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_data.to_csv("courses_train_dataset_v2.csv", index=False)
# Function to generate augmented data for training set
def augment_data(train_data):
    augmented_data = []
    for _, row in train_data.iterrows():
        skills = row['Skills']
        for _ in range(10):
            shuffled_skills = get_random_permutation(skills)
            augmented_row = row.copy()
            augmented_row['Skills'] = shuffled_skills
            augmented_data.append(augmented_row)
    return pd.DataFrame(augmented_data)


# Apply data augmentation only to the training data
augmented_train_data = augment_data(train_data)

# Write augmented training data to CSV
augmented_train_data.to_csv("courses_train_augmented_dataset_v2.csv", index=False)

# Write test data to CSV
test_data.to_csv("courses_test_dataset_v2.csv", index=False)

# Write val data to CSV
val_data.to_csv("courses_validation_dataset_v2.csv", index=False)
