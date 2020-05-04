import pandas as pd

from sklearn.model_selection import train_test_split


original_train_df = pd.read_csv(
    "data/SemEval-2020-Task5/original/train.csv",
    delimiter=",",
)

train_df, dev_df = train_test_split(original_train_df, test_size=0.1)

print(len(train_df))
print(len(dev_df))
print(train_df.head())
print(dev_df.head())
train_df.to_csv("data/SemEval-2020-Task5/random_split/train.csv", index=False)
dev_df.to_csv("data/SemEval-2020-Task5/random_split/test.csv", index=False)