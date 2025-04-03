import os
import re
import emoji
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions
import string
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

DATA_DIR = '../data'
train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')


def prepare_data():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    train_df['data_split_type'] = "train"
    val_df['data_split_type'] = "val"
    test_df['data_split_type'] = "test"
    df = pd.concat([train_df, val_df, test_df])
    df.reset_index(drop=True, inplace=True)

    df['sentiment'] = df['customer_sentiment'].map({'positive': 0, 'neutral': 1, 'negative': 2})

    df['text_cleaned'] = df.conversation
    df['text_cleaned'] = df['text_cleaned'].str.lower()
    df['text_cleaned'] = df['text_cleaned'].apply(remove_full_pattern)
    df['text_cleaned'] = df['text_cleaned'].str.replace(r"\b(Agent:|Customer:)\s*", "", regex=True)
    df['text_cleaned'] = df['text_cleaned'].apply(lambda text: re.sub(r"\S+@\S+|www\.\S+\.com", "", text))
    df['text_cleaned'] = df['text_cleaned'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['text_cleaned'] = df['text_cleaned'].apply(lambda x: re.sub(r'\d+', '', x))
    df['text_cleaned'] = df['text_cleaned'].apply(lambda x: ' '.join(x.split()))
    df['text_cleaned'] = df['text_cleaned'].apply(lambda x: re.sub(r'(\W)\1+', r'\1', x))
    df['text_cleaned'] = df['text_cleaned'].apply(convert_emojis_to_words)
    df['text_cleaned'] = df['text_cleaned'].apply(lambda x: re.sub(r"[^\w\s]", '', x))
    df['text_cleaned'] = df['text_cleaned'].apply(lambda x: contractions.fix(x))

    stop_words_set = set(stopwords.words('english'))
    no_stopwords = []
    for sentence in df["text_cleaned"]:
        no_stopwords.append(' '.join(word for word in sentence.split() if word not in stop_words_set))
    df["text_cleaned"] = no_stopwords

    df["text_cleaned"] = df["text_cleaned"].apply(lambda text: lemmatize_words(text))

    df.to_csv(os.path.join(os.path.dirname(__file__), "../final/merged.csv"), index=False)

    return df


def convert_emojis_to_words(text):
    converted_text = emoji.demojize(text)
    return converted_text


def remove_full_pattern(text):
    full_pattern = r"Thank you for calling BrownBox Customer Support\. My name is \w+\. How may I assist you today\?"
    text = re.sub(full_pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
