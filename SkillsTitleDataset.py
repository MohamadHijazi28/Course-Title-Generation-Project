from torch.utils.data import Dataset


class SkillsTitleDataset(Dataset):
    def __init__(self, data, tokenizer, test=False, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Formatting the data
        course_skills = self.data.iloc[idx]['Skills']
        course_title = self.data.iloc[idx]['Title']
        course_skills = f"[Skills] {course_skills} [Title]"
        # Tokenize inputs and labels
        inputs = self.tokenizer.encode_plus(
            course_skills,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer.encode_plus(
            course_title,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels['input_ids'].flatten(),
        }
