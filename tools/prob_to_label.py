import pandas as pd
import numpy as np


def prob_to_label(input_path, output_path):
    input_df = pd.read_csv(input_path)
    pred_label = np.round(input_df['pred_prob'].to_numpy()).astype(int)
    input_df.columns = ['sentenceID','pred_label']
    input_df['pred_label'] = pred_label
    input_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_path = "output/SemEval-2020-Task5/roberta-large/finetuned_with_pseudo_labels/final_model/seq_128_batch_8_epochs_3.0_lr_2e-05/submission_stacking.csv"
    output_path = "subtask1_roberta_large_with_pseudo_labels.csv"
    prob_to_label(input_path, output_path)