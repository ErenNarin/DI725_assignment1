import torch
from torch.utils.data import Dataset


class CustomerSentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256, text_column='text_cleaned', sentiment_column='sentiment', ):
        self.data = df
        self.texts = self.data[text_column].values
        self.labels = self.data[sentiment_column].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Use the tokenizer's method to encode the text
        input_ids = self.tokenizer.encode(text)

        # Ensure the sequence is at most max_length
        input_ids = input_ids[:self.max_length]

        # Padding if necessary to ensure all sequences are of the same length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            # Append zeros at the end for padding
            input_ids = input_ids + [0] * padding_length

        # Ensure it returns a torch tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids, torch.tensor(label, dtype=torch.long)
