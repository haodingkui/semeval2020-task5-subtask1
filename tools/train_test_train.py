import pandas as pd
import numpy as np


train_df = pd.read_csv("data/SemEval-2020-Task5/original/train.csv")
test_df = pd.read_csv("data/SemEval-2020-Task5/original/subtask1_test.csv")
trial_df = pd.read_csv("data/SemEval-2020-Task5/original/trial.csv")

train_sentences = train_df['sentence'].tolist()
print(len(train_sentences))
test_sentences = test_df['sentence'].tolist()
trial_sentences = trial_df['sentence'].tolist()

def extra_same_elem(lst, *lsts):
    iset = set(lst)
    for li in lsts:
        s = set(li)
        iset = iset.intersection(s)
    return list(iset)

lst_train_trial = extra_same_elem(train_sentences, trial_sentences)
lst_trial_test = extra_same_elem(test_sentences, trial_sentences)
print(lst_trial_test)
df = trial_df[trial_df['sentence'].isin(lst_trial_test)]
df.to_csv("trial_test.csv", index=False)
# print(len(lst_train_trial))
print(len(lst_trial_test))