""" Compute the average results of cross validation """

import os
import argparse
from collections import defaultdict

import sys
sys.path.append("/users5/dkhao/github/nlpexperiments-text-classification/")


def get_cross_validation_results_dir(args):
    hyper_param_prefix = f'seq_{args.max_seq_length}_batch_{args.train_batch_size}_epochs_{args.num_train_epochs}_lr_{args.learning_rate}'
    cross_validation_results_dir = os.path.join(hyper_param_prefix, "cross_validation_results")
    return cross_validation_results_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument("--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training."
    )
    parser.add_argument("--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument("--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument("--output_dir",
        type=str,
        default="output/SemEval-2020-Task5/bert-base-cased/",
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument("--num_folds",
        type=int,
        default=10,
        help="The number of folds used for cross validation."
    )
    parser.add_argument("--full_path",
        type=str,
        default=None,
        help="If running manually, this will be used as the full path to the folder."
    )

    args = parser.parse_args()

    cross_validation_results_dir = get_cross_validation_results_dir(args)
    if args.full_path:
        full_path = args.full_path
    else:
        full_path = os.path.join(args.output_dir, cross_validation_results_dir)

    results = defaultdict(float)

    for fold_result_file in os.listdir(full_path):
        if fold_result_file[-4:] != '.txt':
            continue

        with open(os.path.join(full_path, fold_result_file), 'r') as f:
            for line in f.readlines():
                key, _, value = line.split()
                results[key] += float(value)

    # Average results over k-folds and save to file
    for key in results.keys():
        results[key] /= args.num_folds

    # Get overall f1 score.
    precision, recall = results['precision'], results['recall']
    results['f1'] = 2 * (precision * recall) / (precision + recall)

    with open(os.path.join(full_path, 'final.txt'), 'w') as f:
        for key in sorted(results.keys()):
            value = results[key]
            f.write(f'{key} = {value}\n')
    

if __name__ == "__main__":
    main()