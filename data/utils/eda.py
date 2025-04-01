import os
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

train_file = os.path.join(os.path.dirname(__file__), '../train.csv')
test_file = os.path.join(os.path.dirname(__file__), '../test.csv')


def load_data():
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    print("Train Dataset Info:")
    df_train.info()
    print("\nTest Dataset Info:")
    df_test.info()

    return df_train, df_test


def plot_sentiment_distribution(df, target):
    fig = plt.figure(figsize=(6, 6))
    sns.countplot(data=df, x=target, palette="coolwarm")
    plt.title(f"Distribution of {target}")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.show()
    wandb.log({f"Distribution of {target}": wandb.Image(fig)})


def _chi_square_test(df, feature, target):
    contingency_table = pd.crosstab(df[feature], df[target])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    return chi2, p


def plot_feature_importance(df, target):
    categorical_features = df.columns.drop([target])
    correlation_results = {}
    for col in categorical_features:
        chi2, p = _chi_square_test(df, col, target)
        correlation_results[col] = {'Chi2 Score': chi2, 'P-Value': p}

    influential_params = sorted(correlation_results.items(), key=lambda x: x[1]['Chi2 Score'], reverse=True)
    print(f"\nMost Influential Parameters on {target}:")
    for param, values in influential_params:
        print(f"{param}: Chi2 Score = {values['Chi2 Score']:.2f}, P-Value = {values['P-Value']:.5f}")

    fig = plt.figure(figsize=(10, 10))
    params = [param[0] for param in influential_params]
    chi2_scores = [param[1]['Chi2 Score'] for param in influential_params]
    sns.barplot(x=params, y=chi2_scores, palette="coolwarm")
    plt.xticks(rotation=45, ha='right')
    plt.title("Feature Importance Based on Chi-Square Scores")
    plt.xlabel("Features")
    plt.ylabel("Chi-Square Score")
    plt.show()
    wandb.log({"Feature Importance Based on Chi-Square Scores": wandb.Image(fig)})


def plot_sentiment_distribution_by_feature(df, features, target):
    for col in features:
        fig = plt.figure(figsize=(8, 8))
        sns.countplot(data=df, x=col, hue=target, palette="coolwarm")
        plt.title(f"Sentiment Distribution by {col}")
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title='Sentiment')
        plt.show()
        wandb.log({f"Sentiment Distribution by {col}": wandb.Image(fig)})
