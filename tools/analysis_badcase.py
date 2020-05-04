import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix


def output_false_negative(preds, true_label):
    index = np.arange(11700,13000)
    print(index[preds != true_label])
    false_index = index[preds != true_label].tolist()
    print(false_index)
    train_csv = pd.read_csv("data/SemEval-2020-Task5/original/train.csv")
    # print(train_csv.head())
    false_data = train_csv.loc[false_index, :]
    false_data.to_csv('false_data.csv',index=False)
    print(false_data)
    print()

def analysis_pred_results_on_trial():
    trial_submission = pd.read_csv("data/SemEval-2020-Task5/original/trial.csv")

    submission_bert_base_cased= pd.read_csv("random_split_output/SemEval-2020-Task5/bert-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial.csv")
    submission_bert_large_cased = pd.read_csv("random_split_output/SemEval-2020-Task5/bert-large-cased/seq_128_batch_12_epochs_3.0_lr_2e-05/submission_stacking_trial.csv")
    submission_custombert_large_cased = pd.read_csv("random_split_output/SemEval-2020-Task5/custombert-large-cased/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial.csv")
    submission_roberta_base = pd.read_csv("random_split_output/SemEval-2020-Task5/roberta-base/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial.csv")
    submission_roberta_large = pd.read_csv("random_split_output/SemEval-2020-Task5/roberta-large/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial.csv")
    submission_xlnet_base_cased = pd.read_csv("random_split_output/SemEval-2020-Task5/xlnet-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial.csv")
    submission_stacking = pd.read_csv("trial_stacking_result.csv")

    true_label = trial_submission['gold_label'].to_numpy()
    bert_base_cased_preds = np.round(submission_bert_base_cased['pred_prob'].to_numpy())
    bert_large_cased_preds = np.round(submission_bert_large_cased['pred_prob'].to_numpy())
    custombert_large_cased_preds = np.round(submission_custombert_large_cased['pred_prob'].to_numpy())
    roberta_base_preds = np.round(submission_roberta_base['pred_prob'].to_numpy())
    roberta_large_preds = np.round(submission_roberta_large['pred_prob'].to_numpy())
    xlnet_base_cased_preds = np.round(submission_xlnet_base_cased['pred_prob'].to_numpy())
    stacking_preds = submission_stacking['pred_label'].to_numpy()

    print('bert-base-cased')
    print(classification_report(true_label,bert_base_cased_preds, digits=4))
    print("bert-large-cased")
    print(classification_report(true_label,bert_large_cased_preds, digits=4))
    print("custombert-large-cased")
    print(classification_report(true_label,custombert_large_cased_preds, digits=4))
    print("roberta-base")
    print(classification_report(true_label,roberta_base_preds, digits=4))
    print("roberta-large")
    print(classification_report(true_label,roberta_large_preds, digits=4))
    print("xlnet-base-cased")
    print(classification_report(true_label,xlnet_base_cased_preds, digits=4))
    print("stacking")
    print(classification_report(true_label,stacking_preds, digits=4))


def analysis_pred_results_on_trial_test_final():
    trial_submission = pd.read_csv("data/SemEval-2020-Task5/random_split/trial_test.csv")

    submission_bert_base_cased= pd.read_csv("output/SemEval-2020-Task5/bert-base-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_bert_large_cased = pd.read_csv("output/SemEval-2020-Task5/bert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_custombert_large_cased = pd.read_csv("output/SemEval-2020-Task5/custombert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_custombert_large_uncased = pd.read_csv("output/SemEval-2020-Task5/custombert-large-uncased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_roberta_base = pd.read_csv("output/SemEval-2020-Task5/roberta-base/finetuned_with_pseudo_labels/final_model/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_roberta_large = pd.read_csv("output/SemEval-2020-Task5/roberta-large/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    # submission_xlnet_base_cased = pd.read_csv("random_split_output/SemEval-2020-Task5/xlnet-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_stacking = pd.read_csv("trial_test_stacking_result.csv")

    true_label = trial_submission['gold_label'].to_numpy()
    bert_base_cased_preds = np.round(submission_bert_base_cased['pred_prob'].to_numpy())
    bert_large_cased_preds = np.round(submission_bert_large_cased['pred_prob'].to_numpy())
    custombert_large_cased_preds = np.round(submission_custombert_large_cased['pred_prob'].to_numpy())
    custombert_large_uncased_preds = np.round(submission_custombert_large_uncased['pred_prob'].to_numpy())
    roberta_base_preds = np.round(submission_roberta_base['pred_prob'].to_numpy())
    roberta_large_preds = np.round(submission_roberta_large['pred_prob'].to_numpy())
    # xlnet_base_cased_preds = np.round(submission_xlnet_base_cased['pred_prob'].to_numpy())
    stacking_preds = submission_stacking['pred_label'].to_numpy()

    print('bert-base-cased')
    print(classification_report(true_label,bert_base_cased_preds, digits=4))
    print("bert-large-cased")
    print(classification_report(true_label,bert_large_cased_preds, digits=4))
    print("custombert-large-cased")
    print(classification_report(true_label,custombert_large_cased_preds, digits=4))
    print("custombert-large-uncased")
    print(classification_report(true_label,custombert_large_uncased_preds, digits=4))
    print("roberta-base")
    print(classification_report(true_label,roberta_base_preds, digits=4))
    print("roberta-large")
    print(classification_report(true_label,roberta_large_preds, digits=4))
    # print("xlnet-base-cased")
    # print(classification_report(true_label,xlnet_base_cased_preds, digits=4))
    print("stacking")
    print(classification_report(true_label,stacking_preds, digits=4))

def analysis_pred_results_on_trial_test():
    trial_submission = pd.read_csv("data/SemEval-2020-Task5/random_split/trial_test.csv")

    submission_bert_base_cased= pd.read_csv("random_split_output/SemEval-2020-Task5/bert-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_bert_large_cased = pd.read_csv("random_split_output/SemEval-2020-Task5/bert-large-cased/seq_128_batch_12_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_custombert_large_cased = pd.read_csv("random_split_output/SemEval-2020-Task5/custombert-large-cased/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_roberta_base = pd.read_csv("random_split_output/SemEval-2020-Task5/roberta-base/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_roberta_large = pd.read_csv("random_split_output/SemEval-2020-Task5/roberta-large/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_xlnet_base_cased = pd.read_csv("random_split_output/SemEval-2020-Task5/xlnet-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking_trial_test.csv")
    submission_stacking = pd.read_csv("trial_stacking_result.csv")

    true_label = trial_submission['gold_label'].to_numpy()
    bert_base_cased_preds = np.round(submission_bert_base_cased['pred_prob'].to_numpy())
    bert_large_cased_preds = np.round(submission_bert_large_cased['pred_prob'].to_numpy())
    custombert_large_cased_preds = np.round(submission_custombert_large_cased['pred_prob'].to_numpy())
    roberta_base_preds = np.round(submission_roberta_base['pred_prob'].to_numpy())
    roberta_large_preds = np.round(submission_roberta_large['pred_prob'].to_numpy())
    xlnet_base_cased_preds = np.round(submission_xlnet_base_cased['pred_prob'].to_numpy())
    stacking_preds = submission_stacking['pred_label'].to_numpy()

    print('bert-base-cased')
    print(classification_report(true_label,bert_base_cased_preds, digits=4))
    print("bert-large-cased")
    print(classification_report(true_label,bert_large_cased_preds, digits=4))
    print("custombert-large-cased")
    print(classification_report(true_label,custombert_large_cased_preds, digits=4))
    print("roberta-base")
    print(classification_report(true_label,roberta_base_preds, digits=4))
    print("roberta-large")
    print(classification_report(true_label,roberta_large_preds, digits=4))
    # print("xlnet-base-cased")
    # print(classification_report(true_label,xlnet_base_cased_preds, digits=4))
    print("stacking")
    print(classification_report(true_label,stacking_preds, digits=4))

def analysis_pred_results():
    sample_submission = pd.read_csv("data/SemEval-2020-Task5/random_split/dev.csv")

    submission_bert_base_cased= pd.read_csv("random_split_output/SemEval-2020-Task5/bert-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking.csv")
    submission_bert_large_cased = pd.read_csv("random_split_output/SemEval-2020-Task5/bert-large-cased/seq_128_batch_12_epochs_3.0_lr_2e-05/submission_stacking.csv")
    submission_custombert_large_cased = pd.read_csv("random_split_output/SemEval-2020-Task5/custombert-large-cased/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv")
    submission_roberta_base = pd.read_csv("random_split_output/SemEval-2020-Task5/roberta-base/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking.csv")
    submission_roberta_large = pd.read_csv("random_split_output/SemEval-2020-Task5/roberta-large/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv")
    submission_xlnet_base_cased = pd.read_csv("random_split_output/SemEval-2020-Task5/xlnet-base-cased/seq_128_batch_32_epochs_3.0_lr_2e-05/submission_stacking.csv")
    submission_stacking = pd.read_csv("test_stacking_result.csv")

    true_label = sample_submission['gold_label'].to_numpy()
    bert_base_cased_preds = np.round(submission_bert_base_cased['pred_prob'].to_numpy())
    bert_large_cased_preds = np.round(submission_bert_large_cased['pred_prob'].to_numpy())
    custombert_large_cased_preds = np.round(submission_custombert_large_cased['pred_prob'].to_numpy())
    roberta_base_preds = np.round(submission_roberta_base['pred_prob'].to_numpy())
    roberta_large_preds = np.round(submission_roberta_large['pred_prob'].to_numpy())
    xlnet_base_cased_preds = np.round(submission_xlnet_base_cased['pred_prob'].to_numpy())
    
    stacking_preds = submission_stacking['pred_label'].to_numpy()

    print('bert-base-cased')
    print(classification_report(true_label,bert_base_cased_preds, digits=4))
    print("bert-large-cased")
    print(classification_report(true_label,bert_large_cased_preds, digits=4))
    print("custombert-large-cased")
    print(classification_report(true_label,custombert_large_cased_preds, digits=4))
    print("roberta-base")
    print(classification_report(true_label,roberta_base_preds, digits=4))
    print("roberta-large")
    print(classification_report(true_label,roberta_large_preds, digits=4))
    # print("xlnet-base-cased")
    # print(classification_report(true_label,xlnet_base_cased_preds, digits=4))
    print("stacking")
    print(classification_report(true_label,stacking_preds, digits=4))



if __name__ == "__main__":
    # sample_submission = pd.read_csv("submissions/sample_submission.csv")
    # submission_roberta = pd.read_csv("submissions/_w3_submission_roberta_large.csv")
    # submission_vote_weighted = pd.read_csv("submissions/submission_vote_weighted.csv")

    # true_label = sample_submission['pred_label'].to_numpy()
    # roberta_preds = submission_roberta['pred_label'].to_numpy()
    # vote_weighted_preds = submission_vote_weighted['pred_label'].to_numpy()

    # output_false_negative(roberta_preds, true_label)
    # output_false_negative(vote_weighted_preds, true_label)

    # analysis_pred_results()
    # analysis_pred_results_on_trial_test()
    # analysis_pred_results_on_trial()
    analysis_pred_results_on_trial_test_final()