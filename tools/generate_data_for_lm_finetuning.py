import pandas as pd
import numpy as np


def generate_line_by_line_data_train():
    train_df = pd.read_csv(
        "data/SemEval-2020-Task5/original/train.csv", 
        delimiter=',', 
        header=0)
    train_sentence_df = train_df['sentence']
    print(len(train_sentence_df))
    train_sentence_df.to_csv("data/SemEval-2020-Task5/lm_finetuning_train_sentences.csv", index=False)


def generate_line_by_line_data_test():
    test_df = pd.read_csv(
        "data/SemEval-2020-Task5/original/subtask1_test.csv", 
        delimiter=',', 
        header=0)
    test_sentence_df = test_df['sentence']
    print(len(test_sentence_df))
    test_sentence_df.to_csv("data/SemEval-2020-Task5/lm_finetuning_test_sentences.csv", index=False)


def generate_line_by_line_data_train_test():
    train_df = pd.read_csv(
        "data/SemEval-2020-Task5/original/train.csv", 
        delimiter=',', 
        header=0)
    test_df = pd.read_csv(
        "data/SemEval-2020-Task5/original/subtask1_test.csv", 
        delimiter=',', 
        header=0)
    train_sentence_df = train_df['sentence']
    test_sentence_df = test_df['sentence']
    all_sentence_df = train_sentence_df.append(test_sentence_df)
    print(len(all_sentence_df))
    all_sentence_df.to_csv("data/SemEval-2020-Task5/lm_finetuning_all_sentences.csv", index=False)


if __name__ == "__main__":
    generate_line_by_line_data_train()
    # generate_line_by_line_data_test()
    # generate_line_by_line_data_train_test()