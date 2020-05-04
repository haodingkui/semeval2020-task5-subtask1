import os
import glob
import logging

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression


logger = logging.getLogger(__name__)

MODELS = [
    # "bert-base-cased",
    # "bert-large-cased",
    "custombert-large-cased",
    "xlnet-base-cased",
    "xlnet-large-cased",
    # "roberta-base",
    "roberta-large",
]

MODEL_CV_RESULT_DIRS = {
    "bert-base-cased": "output/SemEval-2020-Task5/bert-base-cased/finetuned_with_pseudo_labels/seq_128_batch_32_epochs_3.0_lr_2e-05/cross_validation_results",
    "bert-large-cased": "output/SemEval-2020-Task5/bert-large-cased/finetuned_with_pseudo_labels/seq_128_batch_12_epochs_3.0_lr_2e-05/cross_validation_results",
    "custombert-large-cased": "output/SemEval-2020-Task5/custombert-large-cased/finetuned_with_pseudo_labels/seq_128_batch_8_epochs_3.0_lr_2e-05/cross_validation_results",
    "xlnet-base-cased": "output/SemEval-2020-Task5/xlnet-base-cased/finetuned_with_pseudo_labels/seq_128_batch_32_epochs_3.0_lr_2e-05/cross_validation_results",
    "xlnet-large-cased": "output/SemEval-2020-Task5/xlnet-large-cased/finetuned_with_pseudo_labels/seq_128_batch_8_epochs_3.0_lr_2e-05/cross_validation_results",
    "roberta-base": "output/SemEval-2020-Task5/roberta-base/finetuned_with_pseudo_labels/seq_128_batch_32_epochs_3.0_lr_2e-05/cross_validation_results",
    "roberta-large": "output/SemEval-2020-Task5/roberta-large/finetuned_with_pseudo_labels/seq_128_batch_8_epochs_3.0_lr_2e-05/cross_validation_results",
}

MODEL_TEST_RESULT_PATHS = {
    "bert-base-cased": "output/SemEval-2020-Task5/bert-base-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "bert-large-cased": "output/SemEval-2020-Task5/bert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "custombert-large-cased": "output/SemEval-2020-Task5/custombert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "xlnet-base-cased": "output/SemEval-2020-Task5/xlnet-base-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "xlnet-large-cased": "output/SemEval-2020-Task5/xlnet-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "roberta-base": "output/SemEval-2020-Task5/roberta-base/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "roberta-large": "output/SemEval-2020-Task5/roberta-large/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv",
}

MODEL_TRIAL_RESULT_PATHS = {
    "bert-base-cased": "output/SemEval-2020-Task5/bert-base-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "bert-large-cased": "output/SemEval-2020-Task5/bert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "custombert-large-cased": "output/SemEval-2020-Task5/custombert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "xlnet-base-cased": "output/SemEval-2020-Task5/xlnet-base-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "xlnet-large-cased": "output/SemEval-2020-Task5/xlnet-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "roberta-base": "output/SemEval-2020-Task5/roberta-base/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "roberta-large": "output/SemEval-2020-Task5/roberta-large/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
}

TRAIN_DIR = "data/SemEval-2020-Task5/original/train.csv"
SAMPLE_SUBMISSION_PATH = "submissions/test_sample_submission.csv"
TRIAL_SAMPLE_SUBMISSION_PATH = "output/SemEval-2020-Task5/bert-base-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv"

NUM_TRAIN_EXAMPLES = 13000
NUM_TEST_EXAMPLES = 7000
NUM_TRIAL_EXAMPLES = 910

NUM_FOLDS = 10


def trial():
    logger.info("Creating train and test sets for blending.")

    train_df = pd.read_csv(TRAIN_DIR, delimiter=",")

    train_labels = train_df['gold_label'].to_numpy()

    train_prob_features = np.zeros((NUM_TRAIN_EXAMPLES, len(MODELS)))
    test_prob_features = np.zeros((NUM_TRIAL_EXAMPLES, len(MODELS)))

    for j, model_name in enumerate(MODELS):
        print(model_name)
        cross_validation_results_dir = MODEL_CV_RESULT_DIRS[model_name]
        model_outputs = glob.glob(cross_validation_results_dir +'/*.csv')
        model_outputs.sort(key=lambda str: int(str[:-4].split("_")[-1]))
        idx_counter = 0
        for fold_result_file in model_outputs:
            print(fold_result_file)
            fold_result_df = pd.read_csv(fold_result_file, delimiter=",")

            # fold_idxs = fold_result_df['sentenceID'].to_numpy() - 100000 * np.ones(len(fold_result_df))
            # fold_idxs = fold_idxs.astype(int)
            fold_probs = fold_result_df['pred_prob'].to_numpy()
            # train_prob_features[fold_idxs, j] = fold_probs
            fold_probs_len = len(fold_probs)
            fold_idxs = np.arange(idx_counter,idx_counter+fold_probs_len)
            idx_counter += fold_probs_len
            train_prob_features[fold_idxs, j] = fold_probs
        test_result_file = MODEL_TRIAL_RESULT_PATHS[model_name]
        test_result_df = pd.read_csv(test_result_file, delimiter=",")
        test_prob_features[:, j] = test_result_df['pred_prob']

    clf = LogisticRegression()
    clf.fit(train_prob_features, train_labels)
    test_prob = clf.predict_proba(test_prob_features)[:,1]

    print("Linear stretch of predictions to [0,1]")
    test_prob = (test_prob - test_prob.min()) / (test_prob.max() - test_prob.min())
    print(len(test_prob))
    print()
    test_pred_labels = np.round(test_prob).astype(int)

    submission_sample_df = pd.read_csv(TRIAL_SAMPLE_SUBMISSION_PATH, delimiter=",")
    submission_sample_df.columns = ['sentenceID','pred_label']
    submission_sample_df['pred_label'] = test_pred_labels
    print(submission_sample_df.head())
    submission_sample_df.to_csv("trial_test_stacking_result.csv", index=False)

def main():
    logger.info("Creating train and test sets for blending.")

    train_df = pd.read_csv(TRAIN_DIR, delimiter=",")

    train_labels = train_df['gold_label'].to_numpy()

    train_prob_features = np.zeros((NUM_TRAIN_EXAMPLES, len(MODELS)))
    test_prob_features = np.zeros((NUM_TEST_EXAMPLES, len(MODELS)))

    for j, model_name in enumerate(MODELS):
        print(model_name)
        cross_validation_results_dir = MODEL_CV_RESULT_DIRS[model_name]
        model_outputs = glob.glob(cross_validation_results_dir +'/*.csv')
        model_outputs.sort(key=lambda str: int(str[:-4].split("_")[-1]))
        idx_counter = 0
        for fold_result_file in model_outputs:
            print(fold_result_file)
            fold_result_df = pd.read_csv(fold_result_file, delimiter=",")

            # fold_idxs = fold_result_df['sentenceID'].to_numpy() - 100000 * np.ones(len(fold_result_df))
            # fold_idxs = fold_idxs.astype(int)
            fold_probs = fold_result_df['pred_prob'].to_numpy()
            # train_prob_features[fold_idxs, j] = fold_probs
            fold_probs_len = len(fold_probs)
            fold_idxs = np.arange(idx_counter,idx_counter+fold_probs_len)
            idx_counter += fold_probs_len
            train_prob_features[fold_idxs, j] = fold_probs
        test_result_file = MODEL_TEST_RESULT_PATHS[model_name]
        test_result_df = pd.read_csv(test_result_file, delimiter=",")
        test_prob_features[:, j] = test_result_df['pred_prob']

    clf = LogisticRegression()
    clf.fit(train_prob_features, train_labels)
    test_prob = clf.predict_proba(test_prob_features)[:,1]

    print("Linear stretch of predictions to [0,1]")
    test_prob = (test_prob - test_prob.min()) / (test_prob.max() - test_prob.min())
    print(len(test_prob))
    print()
    test_pred_labels = np.round(test_prob).astype(int)

    submission_sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH, delimiter=",")
    submission_sample_df.columns = ['sentenceID','pred_label']
    submission_sample_df['pred_label'] = test_pred_labels
    print(submission_sample_df.head())
    submission_sample_df.to_csv("test_stacking_result.csv", index=False)


if __name__ == "__main__":
    # main()
    trial()