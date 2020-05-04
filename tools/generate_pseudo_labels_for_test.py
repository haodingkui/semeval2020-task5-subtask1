import pandas as pd
import numpy as np


def main():
    original_test_df = pd.read_csv(
        "data/SemEval-2020-Task5/original/subtask1_test.csv",
        delimiter=",",
    )
    # print(original_test.head())
    roberta_large_test_submission = pd.read_csv(
        "output/SemEval-2020-Task5/roberta-large/seq_128_batch_8_epochs_3.0_lr_2e-05/submission.csv", 
        delimiter=',', 
        header=0)
    print(roberta_large_test_submission.head())
    xlnet_large_cased_test_submission = pd.read_csv(
        "output/SemEval-2020-Task5/xlnet-large-cased/seq_128_batch_8_epochs_3.0_lr_2e-05/submission.csv", 
        delimiter=',', 
        header=0)
    print(xlnet_large_cased_test_submission.head())
    bert_large_cased_test_submission = pd.read_csv(
        "output/SemEval-2020-Task5/bert-large-cased/seq_128_batch_12_epochs_3.0_lr_2e-05/submission.csv", 
        delimiter=',', 
        header=0)
    custombert_large_cased_test_submission = pd.read_csv(
        "output/SemEval-2020-Task5/custombert-large-cased/seq_128_batch_8_epochs_3.0_lr_2e-05/submission.csv", 
        delimiter=',', 
        header=0)

    roberta_large_pred_label = roberta_large_test_submission['pred_label'].to_numpy()
    xlnet_large_cased_pred_label = xlnet_large_cased_test_submission['pred_label'].to_numpy()
    bert_large_cased_pred_label = bert_large_cased_test_submission['pred_label'].to_numpy()
    custombert_large_cased_pred_label = custombert_large_cased_test_submission['pred_label'].to_numpy()
    index1 = np.arange(0, len(roberta_large_pred_label))
    index_rx = index1[roberta_large_pred_label==xlnet_large_cased_pred_label]
    index_rb = index1[roberta_large_pred_label==bert_large_cased_pred_label]
    index_rc = index1[roberta_large_pred_label==custombert_large_cased_pred_label]
    # print(index_rx)
    # print(index_rb)
    
    print(len(index_rx))
    print(len(index_rb))
    print(len(index_rc))
    print(len(np.intersect1d(index_rx, index_rb)))

    intersect_rxb = np.intersect1d(index_rx, index_rb)
    intersect_rxbc =np.intersect1d(intersect_rxb, index_rc)
    print(len(intersect_rxbc))
    intersect_rxbc_df = pd.DataFrame(intersect_rxbc)
    intersect_rxbc_df.to_csv("intersect.csv", index=False, header=False)

    
    intersect_submission_df = roberta_large_test_submission.loc[intersect_rxbc,["sentenceID","pred_label"]]
    pseudo_labeled_test_df = original_test_df.join(
        intersect_submission_df.set_index("sentenceID"), 
        on="sentenceID",
        how="inner"
    )
    pseudo_labeled_test_df = pseudo_labeled_test_df[['sentenceID', "pred_label", "sentence"]]
    pseudo_labeled_test_df.columns = ['sentenceID', "gold_label", "sentence"]
    print(pseudo_labeled_test_df.head())
    print(len(pseudo_labeled_test_df))
    pseudo_labeled_test_df.to_csv("pseudo_labeled_test.csv", index=False)

def main1():
    original_test_df = pd.read_csv(
        "data/SemEval-2020-Task5/original/subtask1_test.csv",
        delimiter=",",
    )
    # print(original_test.head())
    roberta_large_test_submission = pd.read_csv(
        "output/SemEval-2020-Task5/roberta-large/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv", 
        delimiter=',', 
        header=0)
    rounded_roberta_pred_prob = np.round(roberta_large_test_submission['pred_prob'].to_numpy()).astype(int)
    roberta_large_test_submission['pred_prob'] = rounded_roberta_pred_prob
    roberta_large_test_submission.columns = ['sentenceID', 'pred_label']
    print(roberta_large_test_submission.head())
    
    xlnet_large_cased_test_submission = pd.read_csv(
        "output/SemEval-2020-Task5/xlnet-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv", 
        delimiter=',', 
        header=0)
    rounded_xlnet_pred_prob = np.round(xlnet_large_cased_test_submission['pred_prob'].to_numpy()).astype(int)
    xlnet_large_cased_test_submission['pred_prob'] = rounded_xlnet_pred_prob
    xlnet_large_cased_test_submission.columns = ['sentenceID', 'pred_label']
    print(xlnet_large_cased_test_submission.head())
    

    bert_large_cased_test_submission = pd.read_csv(
        "output/SemEval-2020-Task5/bert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv", 
        delimiter=',', 
        header=0)
    rounded_bert_pred_prob = np.round(bert_large_cased_test_submission['pred_prob'].to_numpy()).astype(int)
    bert_large_cased_test_submission['pred_prob'] = rounded_bert_pred_prob
    bert_large_cased_test_submission.columns = ['sentenceID', 'pred_label']

    custombert_large_cased_test_submission = pd.read_csv(
        "output/SemEval-2020-Task5/custombert-large-cased/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv", 
        delimiter=',', 
        header=0)
    rounded_custombert_pred_prob = np.round(custombert_large_cased_test_submission['pred_prob'].to_numpy()).astype(int)
    custombert_large_cased_test_submission['pred_prob'] = rounded_custombert_pred_prob
    custombert_large_cased_test_submission.columns = ['sentenceID', 'pred_label']

    roberta_large_pred_label = roberta_large_test_submission['pred_label'].to_numpy()
    xlnet_large_cased_pred_label = xlnet_large_cased_test_submission['pred_label'].to_numpy()
    custombert_large_cased_pred_label = custombert_large_cased_test_submission['pred_label'].to_numpy()
    bert_large_cased_pred_label = bert_large_cased_test_submission['pred_label'].to_numpy()
    
    index1 = np.arange(0, len(roberta_large_pred_label))
    index_rx = index1[roberta_large_pred_label==xlnet_large_cased_pred_label]
    index_rb = index1[roberta_large_pred_label==bert_large_cased_pred_label]
    index_rc = index1[roberta_large_pred_label==custombert_large_cased_pred_label]
    print(len(index_rx))
    print(len(index_rc))
    
    # print(len(index_rx))
    # print(len(index_rb))
    # print(len(index_rc))
    # print(len(np.intersect1d(index_rx, index_rb)))

    intersect_rxb = np.intersect1d(index_rx, index_rb)
    intersect_rxbc =np.intersect1d(intersect_rxb, index_rc)
    print(len(intersect_rxbc))
    # intersect_rxbc_df = pd.DataFrame(intersect_rxbc)
    # intersect_rxbc_df.to_csv("intersect.csv", index=False, header=False)

    
    intersect_submission_df = roberta_large_test_submission.loc[intersect_rxbc,["sentenceID","pred_label"]]
    pseudo_labeled_test_df = original_test_df.join(
        intersect_submission_df.set_index("sentenceID"), 
        on="sentenceID",
        how="inner"
    )
    pseudo_labeled_test_df = pseudo_labeled_test_df[['sentenceID', "pred_label", "sentence"]]
    pseudo_labeled_test_df.columns = ['sentenceID', "gold_label", "sentence"]
    print(pseudo_labeled_test_df.head())
    print(len(pseudo_labeled_test_df))

    pseudo_labeled_test_df.to_csv("pseudo_labeled_test_0227.csv", index=False)


if __name__ == "__main__":
    # main()
    main1()