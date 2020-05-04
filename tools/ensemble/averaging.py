import glob

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, precision_recall_curve

MODELS = [
    # "bert-base-cased",
    # "bert-large-cased",
    "custombert-large-cased",
    # "custombert-large-uncased",
    # "xlnet-base-cased",
    "xlnet-large-cased",
    # "roberta-base",
    "roberta-large",
]

MODEL_WEIGHTS = {
    "bert-base-cased": 1,
    "bert-large-cased": 1,
    "custombert-large-cased": 1,
    "custombert-large-uncased": 1,
    "xlnet-base-cased": 1,
    "xlnet-large-cased": 1,
    "roberta-base": 1,
    "roberta-large": 2,
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

MODEL_TRIAL_TEST_RESULT_PATHS = {
    "albert-xxlarge-v2": "output/SemEval-2020-Task5/albert-xxlarge-v2/final_model/seq_128_batch_8_epochs_3.0_lr_3e-05/submission_stacking_trial_test.csv",
    "bert-base-cased": "output/SemEval-2020-Task5/bert-base-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "bert-large-cased": "output/SemEval-2020-Task5/bert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "custombert-large-cased": "output/SemEval-2020-Task5/custombert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "custombert-large-uncased": "output/SemEval-2020-Task5/custombert-large-uncased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "xlnet-base-cased": "output/SemEval-2020-Task5/xlnet-base-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "xlnet-large-cased": "output/SemEval-2020-Task5/xlnet-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "roberta-base": "output/SemEval-2020-Task5/roberta-base/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "roberta-large": "output/SemEval-2020-Task5/roberta-large/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
}

MODEL_TRIAL_TEST_RESULT_PATHS1 = {
    "bert-base-cased": "random_split_output/SemEval-2020-Task5/bert-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "bert-large-cased": "random_split_output/SemEval-2020-Task5/bert-large-cased/seq_128_batch_12_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "custombert-large-cased": "random_split_output/SemEval-2020-Task5/custombert-large-cased/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "xlnet-base-cased": "random_split_output/SemEval-2020-Task5/xlnet-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "xlnet-large-cased": "random_split_output/SemEval-2020-Task5/xlnet-large-cased/seq_128_batch_12_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "roberta-base": "random_split_output/SemEval-2020-Task5/roberta-base/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
    "roberta-large": "random_split_output/SemEval-2020-Task5/roberta-large/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv",
}

MODEL_TRIAL_RESULT_PATHS1 = {
    "bert-base-cased": "random_split_output/SemEval-2020-Task5/bert-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "bert-large-cased": "random_split_output/SemEval-2020-Task5/bert-large-cased/seq_128_batch_12_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "custombert-large-cased": "random_split_output/SemEval-2020-Task5/custombert-large-cased/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "xlnet-base-cased": "random_split_output/SemEval-2020-Task5/xlnet-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "xlnet-large-cased": "random_split_output/SemEval-2020-Task5/xlnet-large-cased/seq_128_batch_12_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "roberta-base": "random_split_output/SemEval-2020-Task5/roberta-base/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "roberta-large": "random_split_output/SemEval-2020-Task5/roberta-large/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
}

MODEL_TRIAL_RESULT_PATHS = {
    "bert-base-cased": "output/SemEval-2020-Task5/bert-base-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "bert-large-cased": "output/SemEval-2020-Task5/bert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "custombert-large-cased": "output/SemEval-2020-Task5/custombert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "custombert-large-uncased": "output/SemEval-2020-Task5/custombert-large-uncased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "xlnet-base-cased": "output/SemEval-2020-Task5/xlnet-base-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "xlnet-large-cased": "output/SemEval-2020-Task5/xlnet-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "roberta-base": "output/SemEval-2020-Task5/roberta-base/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
    "roberta-large": "output/SemEval-2020-Task5/roberta-large/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial.csv",
}


MODEL_DEV_RESULT_PATHS = {
    "bert-base-cased": "random_split_output/SemEval-2020-Task5/bert-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "bert-large-cased": "random_split_output/SemEval-2020-Task5/bert-large-cased/seq_128_batch_12_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "custombert-large-cased": "random_split_output/SemEval-2020-Task5/custombert-large-cased/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "xlnet-base-cased": "random_split_output/SemEval-2020-Task5/xlnet-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "xlnet-large-cased": "random_split_output/SemEval-2020-Task5/xlnet-large-cased/seq_128_batch_12_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "roberta-base": "random_split_output/SemEval-2020-Task5/roberta-base/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking.csv",
    "roberta-large": "random_split_output/SemEval-2020-Task5/roberta-large/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv",
}


MODEL_CV_RESULT_DIRS = {
    "bert-base-cased": "output/SemEval-2020-Task5/bert-base-cased/finetuned_with_pseudo_labels/seq_128_batch_32_epochs_3.0_lr_2e-05/cross_validation_results",
    "bert-large-cased": "output/SemEval-2020-Task5/bert-large-cased/finetuned_with_pseudo_labels/seq_128_batch_12_epochs_3.0_lr_2e-05/cross_validation_results",
    "custombert-large-cased": "output/SemEval-2020-Task5/custombert-large-cased/finetuned_with_pseudo_labels/seq_128_batch_8_epochs_3.0_lr_2e-05/cross_validation_results",
    "xlnet-base-cased": "output/SemEval-2020-Task5/xlnet-base-cased/finetuned_with_pseudo_labels/seq_128_batch_32_epochs_3.0_lr_2e-05/cross_validation_results",
    "xlnet-large-cased": "output/SemEval-2020-Task5/xlnet-large-cased/finetuned_with_pseudo_labels/seq_128_batch_8_epochs_3.0_lr_2e-05/cross_validation_results",
    "roberta-base": "output/SemEval-2020-Task5/roberta-base/finetuned_with_pseudo_labels/seq_128_batch_32_epochs_3.0_lr_2e-05/cross_validation_results",
    "roberta-large": "output/SemEval-2020-Task5/roberta-large/finetuned_with_pseudo_labels/seq_128_batch_8_epochs_3.0_lr_2e-05/cross_validation_results",
}


TRAIN_PATH = "data/SemEval-2020-Task5/original/train.csv"
TEST_SAMPLE_SUBMISSION_PATH = "submissions/test_sample_submission.csv"
TRIAL_TEST_SAMPLE_SUBMISSION_PATH = "output/SemEval-2020-Task5/bert-base-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv"
TRIAL_SAMPLE_SUBMISSION_PATH = "random_split_output/SemEval-2020-Task5/bert-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial.csv"
DEV_SAMPLE_SUBMISSION_PATH = "random_split_output/SemEval-2020-Task5/bert-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking.csv"


def threshold_search(y_true, y_proba):
    precision , recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001) 
    F = 2 / (1/precision + 1/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    return best_th 


def get_model_cv_df(model_name):
    cross_validation_results_dir = MODEL_CV_RESULT_DIRS[model_name]
    model_outputs = glob.glob(cross_validation_results_dir +'/*.csv')
    model_outputs.sort(key=lambda str: int(str[:-4].split("_")[-1]))
    idx_counter = 0
    train_pred_df = None
    for fold_result_file in model_outputs:
        fold_result_df = pd.read_csv(fold_result_file, delimiter=",")
        if train_pred_df is None:
            train_pred_df = fold_result_df
        else:
            train_pred_df =train_pred_df.append(fold_result_df, ignore_index=True)
    print(train_pred_df['sentenceID'])
    return train_pred_df


def averaging_on_train_cv():
    trial_test_prob = None
    for i, model_name in enumerate(MODELS):
        # trial_test_result_file = MODEL_TRIAL_TEST_RESULT_PATHS[model_name]
        # trial_test_result_df = pd.read_csv(trial_test_result_file, delimiter=",")
        trial_test_result_df = get_model_cv_df(model_name)
        trial_test_prob_i = trial_test_result_df['pred_prob'].to_numpy()
        if trial_test_prob is None:
            trial_test_prob = trial_test_prob_i
        else:
            trial_test_prob += trial_test_prob_i
    average_test_prob = trial_test_prob / len(MODELS)
    sample_submission_df = get_model_cv_df("bert-base-cased")
    sample_submission_df['pred_prob'] = average_test_prob
    sample_submission_df.to_csv("average_result.csv", index=False)



def weighted_averaging_on_train_cv():
    trial_test_prob = None
    all_weights = 0
    for i, model_name in enumerate(MODELS):
        model_weight = MODEL_WEIGHTS[model_name]
        # trial_test_result_file = MODEL_TRIAL_TEST_RESULT_PATHS[model_name]
        # trial_test_result_df = pd.read_csv(trial_test_result_file, delimiter=",")
        trial_test_result_df = get_model_cv_df(model_name)
        trial_test_prob_i = trial_test_result_df['pred_prob'].to_numpy()
        all_weights += model_weight
        if trial_test_prob is None:
            trial_test_prob = model_weight * trial_test_prob_i
        else:
            trial_test_prob += model_weight * trial_test_prob_i
    average_test_prob = trial_test_prob / all_weights
    sample_submission_df = get_model_cv_df("bert-base-cased")
    sample_submission_df['pred_prob'] = average_test_prob
    sample_submission_df.to_csv("weighted_average_result.csv", index=False)


def averaging_on_trial_test():
    trial_test_prob = None
    for i, model_name in enumerate(MODELS):
        trial_test_result_file = MODEL_TRIAL_TEST_RESULT_PATHS[model_name]
        trial_test_result_df = pd.read_csv(trial_test_result_file, delimiter=",")
        trial_test_prob_i = trial_test_result_df['pred_prob'].to_numpy()
        if trial_test_prob is None:
            trial_test_prob = trial_test_prob_i
        else:
            trial_test_prob += trial_test_prob_i
    average_test_prob = trial_test_prob / len(MODELS)
    sample_submission_df = pd.read_csv(TRIAL_TEST_SAMPLE_SUBMISSION_PATH)
    sample_submission_df['pred_prob'] = average_test_prob
    sample_submission_df.to_csv("average_result.csv", index=False)

def weighted_averaging_on_trial_test():
    trial_test_prob = None
    all_weights = 0
    for i, model_name in enumerate(MODELS):
        trial_test_result_file = MODEL_TRIAL_TEST_RESULT_PATHS[model_name]
        trial_test_result_df = pd.read_csv(trial_test_result_file, delimiter=",")
        trial_test_prob_i = trial_test_result_df['pred_prob'].to_numpy()
        model_weight = MODEL_WEIGHTS[model_name]
        all_weights += model_weight
        if trial_test_prob is None:
            trial_test_prob = model_weight * trial_test_prob_i
        else:
            trial_test_prob += model_weight * trial_test_prob_i
    
    average_test_prob = trial_test_prob / all_weights
    sample_submission_df = pd.read_csv(TRIAL_TEST_SAMPLE_SUBMISSION_PATH)
    sample_submission_df['pred_prob'] = average_test_prob
    sample_submission_df.to_csv("weighted_average_result.csv", index=False)

def weighted_averaging_on_test():
    trial_test_prob = None
    all_weights = 0
    for i, model_name in enumerate(MODELS):
        trial_test_result_file = MODEL_TEST_RESULT_PATHS[model_name]
        trial_test_result_df = pd.read_csv(trial_test_result_file, delimiter=",")
        trial_test_prob_i = trial_test_result_df['pred_prob'].to_numpy()
        model_weight = MODEL_WEIGHTS[model_name]
        all_weights += model_weight
        if trial_test_prob is None:
            trial_test_prob = model_weight * trial_test_prob_i
        else:
            trial_test_prob += model_weight * trial_test_prob_i
    print("all model weights" + str(all_weights))
    average_test_prob = trial_test_prob / all_weights
    threshold = 0.3437
    average_test_pred_label = (average_test_prob >= threshold).astype(int)

    sample_submission_df = pd.read_csv(TEST_SAMPLE_SUBMISSION_PATH)


    sample_submission_df.columns = ['sentenceID', 'pred_label']
    sample_submission_df['pred_label'] = average_test_pred_label
    sample_submission_df.to_csv("weighted_average_test_result.csv", index=False)


def eval_on_train():
    train_gold_df = pd.read_csv(TRAIN_PATH)

    trial_test_roberta_df = get_model_cv_df("roberta-large")
    trial_test_average_df = pd.read_csv("average_result.csv")
    trial_test_weighted_average_df = pd.read_csv("weighted_average_result.csv")

    gold_label = train_gold_df['gold_label'].to_numpy()
    
    roberta_pred_prob = trial_test_roberta_df['pred_prob'].to_numpy()
    average_pred_prob = trial_test_average_df['pred_prob'].to_numpy()
    weighted_average_pred_prob = trial_test_weighted_average_df['pred_prob'].to_numpy()

    roberta_best_threshold = threshold_search(gold_label, roberta_pred_prob)
    print(roberta_best_threshold)
    average_best_threshold = threshold_search(gold_label, average_pred_prob)
    print(average_best_threshold)
    weighted_average_best_threshold = threshold_search(gold_label, weighted_average_pred_prob)
    print(weighted_average_best_threshold)

    roberta_best_threshold = 0.1
    average_best_threshold = 0.1
    # weighted_average_best_threshold = 0.308
    print(classification_report(gold_label, (roberta_pred_prob>=roberta_best_threshold).astype(int), digits=4))
    print(classification_report(gold_label, (average_pred_prob>=average_best_threshold).astype(int), digits=4))
    print(classification_report(gold_label, (weighted_average_pred_prob>=weighted_average_best_threshold).astype(int), digits=4))


def eval():
    trial_test_gold_df = pd.read_csv("data/SemEval-2020-Task5/random_split/trial_test.csv")
    # trial_gold_df = pd.read_csv("data/SemEval-2020-Task5/original/trial.csv")
    # dev_gold_df = pd.read_csv("data/SemEval-2020-Task5/random_split/dev.csv")
    trial_test_roberta_df = pd.read_csv("output/SemEval-2020-Task5/roberta-large/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    # trial_roberta_df = pd.read_csv("random_split_output/SemEval-2020-Task5/roberta-large/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial.csv")
    # dev_roberta_df = pd.read_csv("random_split_output/SemEval-2020-Task5/roberta-large/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv")
    trial_test_average_df = pd.read_csv("average_result.csv")
    trial_test_weighted_average_df = pd.read_csv("weighted_average_result.csv")
    # trial_test_weighted_average_df = pd.read_csv("weighted_average_test_result.csv")

    gold_label = trial_test_gold_df['gold_label'].to_numpy()
    # gold_label = trial_gold_df['gold_label'].to_numpy()
    # roberta_pred_label = np.round(trial_test_roberta_df['pred_prob'].to_numpy()).astype(int)
    # average_pred_label = np.round(trial_test_average_df['pred_prob'].to_numpy()).astype(int)
    # weighted_average_pred_label = np.round(trial_test_weighted_average_df['pred_prob'].to_numpy()).astype(int)

    roberta_pred_prob = trial_test_roberta_df['pred_prob'].to_numpy()
    average_pred_prob = trial_test_average_df['pred_prob'].to_numpy()
    weighted_average_pred_prob = trial_test_weighted_average_df['pred_prob'].to_numpy()

    roberta_best_threshold = threshold_search(gold_label, roberta_pred_prob)
    print(roberta_best_threshold)
    average_best_threshold = threshold_search(gold_label, average_pred_prob)
    print(average_best_threshold)
    weighted_average_best_threshold = threshold_search(gold_label, weighted_average_pred_prob)
    print(weighted_average_best_threshold)

    
    roberta_best_threshold = 0.1
    average_best_threshold = 0.1
    weighted_average_best_threshold = 0.11
    print(classification_report(gold_label, (roberta_pred_prob>=roberta_best_threshold).astype(int), digits=4))
    print(classification_report(gold_label, (average_pred_prob>=average_best_threshold).astype(int), digits=4))
    print(classification_report(gold_label, (weighted_average_pred_prob>=weighted_average_best_threshold).astype(int), digits=4))
    


if __name__ == "__main__":
    # averaging_on_trial_test()
    # weighted_averaging_on_trial_test()
    # eval()
    weighted_averaging_on_test()

    averaging_on_train_cv()
    weighted_averaging_on_train_cv()
    eval_on_train()