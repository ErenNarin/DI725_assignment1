import torch
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def get_sentiment(model, sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(torch.device("cuda"))
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()

    return reveal_label(prediction)


def reveal_label(label):
    if label == 0:
        return "neutral"
    elif label == 1:
        return "positive"
    else:
        return "negative"


def evaluate_on_test(model, df, max_length=1024):
    preds = []
    labels = []
    try:
        for _, row in df.iterrows():
            prediction = get_sentiment(model, row["text"][:max_length])
            preds.append(prediction)
            labels.append(reveal_label(row["label"]))
            print(f"Prediction: {prediction} - Label: {reveal_label(row['label'])}")
    except Exception as e:
        pass

    return preds, labels
