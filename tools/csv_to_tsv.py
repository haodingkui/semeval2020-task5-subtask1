import pandas as pd
import numpy as np


def csv_to_tsv():
    """ transform original csv format data to tsv. """
    # df = pd.read_csv(
    #     "data/SemEval-2020-Task5/random_split/train.csv", 
    #     delimiter=',', 
    #     header=0)
    # df.to_csv(
    #     "data/SemEval-2020-Task5/random_split/train.tsv",
    #     sep="\t", 
    #     mode="w+",
    #     index=False)
    
    # df = pd.read_csv(
    #     "data/SemEval-2020-Task5/random_split/dev.csv", 
    #     delimiter=',', 
    #     header=0)
    # df.to_csv(
    #     "data/SemEval-2020-Task5/random_split/dev.tsv",
    #     sep="\t", 
    #     mode="w+",
    #     index=False)

    df = pd.read_csv(
        "data/SemEval-2020-Task5/pseudo_labeled_test_0227.csv", 
        delimiter=',', 
        header=0)

    df.to_csv(
        "data/SemEval-2020-Task5/pseudo_labeled_test_0227.tsv",
        sep="\t", 
        mode="w+",
        index=False)


def test_to_submission_sample():
    test_df = pd.read_csv(
        "data/SemEval-2020-Task5/subtask1_test.csv", 
        delimiter=',', 
        header=0)
    test_df.columns = ['sentenceID','pred_label']
    test_df['pred_label'] = np.zeros(len(test_df['pred_label']), dtype=np.int)
    test_df.to_csv("data/SemEval-2020-Task5/test_sample_submission.csv", index=False)


if __name__ == "__main__":
    csv_to_tsv()
    # test_to_submission_sample()