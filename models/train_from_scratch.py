import torch
import wandb
import pickle
import tiktoken
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader
from models.model import GPTConfig
from models.model_nanogpt import BumbleBee
from models.evaluate_from_scratch import plot_confusion_matrix, evaluate
from data.utils.dataset import CustomerSentimentDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train():
    wandb.init(project="DI725_assignment_1_2389088_from_scratch")
    config = wandb.config

    if config.init_from == 'scratch':
        model_config = GPTConfig(vocab_size=config.vocab_size, block_size=config.block_size,
                                 n_layer=config.n_layer, n_head=config.n_head,
                                 n_embd=config.n_embd, dropout=config.dropout)
        model = BumbleBee(model_config).to(device)
    elif str(config.init_from).startswith('gpt'):
        model_config = dict(dropout=config.dropout)
        model = BumbleBee.from_pretrained(model_type=config.init_from, override_args=model_config).to(device)
    else:
        print('Please select the correct model!')

    optimizer = model.configure_optimizers(weight_decay=config.weight_decay,
                                           learning_rate=config.learning_rate,
                                           betas=(config.beta1, config.beta2),
                                           device_type=device)

    df = pd.read_csv("../data/final/merged.csv")

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = CustomerSentimentDataset(df[df.data_split_type == 'train'], max_length=1024, tokenizer=tokenizer)
    test_dataset = CustomerSentimentDataset(df[df.data_split_type == 'test'], max_length=1024, tokenizer=tokenizer)
    val_dataset = CustomerSentimentDataset(df[df.data_split_type == 'val'], max_length=1024, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    best_f1_macro = 0
    best_val_loss = float('inf')
    best_model_path = ""

    for epoch in range(config.epochs):
        model.train()
        # Initialize lists to store batch metrics
        train_losses, train_f1s, train_precs, train_recs, train_accs = [], [], [], [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            # Calculate and store batch metrics
            train_losses.append(loss.item())
            predictions = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            train_f1s.append(f1_score(labels_np, predictions, average='macro'))
            train_precs.append(precision_score(labels_np, predictions, average='macro', zero_division=0))
            train_recs.append(recall_score(labels_np, predictions, average='macro', zero_division=0))
            train_accs.append(accuracy_score(labels_np, predictions))

        # Calculate average metrics over all batches
        avg_train_loss = np.mean(train_losses)
        avg_train_f1 = np.mean(train_f1s)
        avg_train_prec = np.mean(train_precs)
        avg_train_rec = np.mean(train_recs)
        avg_train_acc = np.mean(train_accs)

        model.eval()
        val_loss, val_f1, val_prec, val_rec, val_acc, val_conf_matrix = evaluate(model, val_loader, device)
        print(
            f"\nEpoch {epoch + 1}, Train Loss: {avg_train_loss:.2f}, Train F1 Macro: {avg_train_f1:.2f}, Train Precision Macro: {avg_train_prec:.2f}, Train Recall Macro: {avg_train_rec:.2f}, Train Accuracy: {avg_train_acc:.2f}")
        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss:.2f}, Val F1 Macro: {val_f1:.2f}, Val Precision Macro: {val_prec:.2f}, Val Recall Macro: {val_rec:.2f}, Val Accuracy: {val_acc:.2f}")

        # Saving the best F1 macro result model
        if val_f1 > best_f1_macro:
            best_f1_macro = val_f1
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_config,
                'epoch_num': epoch + 1,
                'best_val_loss': best_val_loss,
                'config': {k: v for k, v in dict(wandb.config).items()},
            }
            state_path = "../models/3_nanoGPT_sentiment_model.pt"
            model_path = '../models/3_nanoGPT_sentiment_model.pkl'
            torch.save(checkpoint, state_path)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Log both training and validation metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss, "train_f1_macro": avg_train_f1, "train_precision_macro": avg_train_prec,
            "train_recall_macro": avg_train_rec, "train_accuracy": avg_train_acc,
            "val_loss": val_loss, "val_f1_macro": val_f1, "val_precision_macro": val_prec, "val_recall_macro": val_rec,
            "val_accuracy": val_acc
        })

    # Load the best model and evaluate on test data
    model.load_state_dict(torch.load(best_model_path)['model'])
    test_loss, test_f1, test_prec, test_rec, test_acc, test_conf_matrix = evaluate(model, test_loader, device)
    print(
        f"\nBest model epoch {checkpoint['epoch_num']} and test results, Test Loss: {test_loss:.2f}, Test F1 Macro: {test_f1:.2f}, Test Precision Macro: {test_prec:.2f}, Test Recall Macro: {test_rec:.2f}, Test Accuracy: {test_acc:.2f}")

    fig = plot_confusion_matrix(test_conf_matrix, class_names=['Positive', 'Neutral', 'Negative'])
    wandb.log({"test_confusion_matrix": wandb.Image(fig)})

    # Log test results to W&B
    wandb.log(
        {"test_loss": test_loss,
         "test_f1_macro": test_f1,
         "test_precision_macro": test_prec,
         "test_recall_macro": test_rec}
    )
    wandb.finish()
