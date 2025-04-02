import os
import re
from transformers import GPT2Tokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_file = os.path.join(os.path.dirname(__file__), '../train.csv')
test_file = os.path.join(os.path.dirname(__file__), '../test.csv')

labels = {"neutral": 0, "positive": 1, "negative": 2}

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def load_reduced_data(columns):
    df_train = pd.read_csv(train_file)[list(columns.keys())].rename(columns=columns)
    df_test = pd.read_csv(test_file)[list(columns.keys())].rename(columns=columns)

    return df_train, df_test


def split_data(df, target, test_size=0.2):
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=42,
                                        stratify=df[target])

    return df_train, df_val


def encode_labels(df, target):
    df_new = df.copy()
    df_new[target] = df_new[target].map(labels)
    df_new[target] = np.array(df_new[target], dtype=np.uint8)

    return df_new


def remove_redundant_lines(text):
    redundant_lines_regex = [
        (
            r"Agent: (Hello|Hi|)(! |, |)Thank you for (calling|contacting) BrownBox Customer Support\. My name is \w+\. How (may|can) I assist you today\?",
            ""),
        (r"\S+@\S+|www\.\S+\.com", "email"),
        ("Agent:", "W:"),  # 'A' is a meaningful word, 'W' is not
        ("Customer:", "C:"),
        ("Specialist:", "S:")
    ]
    for pattern, replace in redundant_lines_regex:
        text = re.sub(pattern, replace, text, flags=re.IGNORECASE)

    return text.strip()


def train_val_split(df, target, test_size=0.2):
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=42, stratify=df[target])

    return df_train, df_val


def tokenize_function(examples, max_length=1024):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)


def save_dataset(df, split):
    df.to_csv(os.path.join(os.path.dirname(__file__), f"../final/{split}.csv"), index=False)


def save_datasets(datasets, split):
    datasets.save_to_disk(f"../data/final/{split}.hf")
