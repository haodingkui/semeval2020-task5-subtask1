import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix


def main():
    avg_prob_df = pd.read_csv("submissions/kaggle_avg.csv")
    roberta_prob_df = pd.read_csv("submissions/submission_roberta_large_prob.csv")
    roberta0_prob_df = pd.read_csv("submissions/submission_roberta_large0_prob.csv")
    roberta_new_prob_df = pd.read_csv("submissions/submission_roberta_large_prob_new.csv")


    trial_test_df = pd.read_csv("data/SemEval-2020-Task5/random_split/trial_test.csv")

    # trial_test_df.sort_values("ID",inplace=True)

    avg_pred_label = np.round(avg_prob_df['pred_prob'].to_numpy()).astype(int)
    roberta_pred_label = np.round(roberta_prob_df['pred_prob'].to_numpy()).astype(int)
    roberta0_pred_label = np.round(roberta0_prob_df['pred_prob'].to_numpy()).astype(int)
    roberta_pred_label_new = np.round(roberta_new_prob_df['pred_prob'].to_numpy()).astype(int)
    gold_label = trial_test_df['gold_label'].to_numpy()
    
    print(gold_label)
    # print(classification_report(gold_label, avg_pred_label, digits=4))
    print(classification_report(gold_label, roberta_pred_label, digits=4))
    print(classification_report(gold_label, roberta0_pred_label, digits=4))
    print(classification_report(gold_label, roberta_pred_label_new, digits=4))
    # print(trial_test_df.head())

if __name__ == "__main__":
    main()