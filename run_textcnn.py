""" Train the TextCNN model for sentence classification. """

import argparse
import logging
import os
import random
import time
import itertools
import math

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import spacy
from torchtext.data import Field, TabularDataset, Dataset, BucketIterator

from models.modeling_textcnn import CNN
from configs.configuration_textcnn import TextCNNConfig
from utils.metrics import acc_precision_recall_f1


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


def load_dataset(train_data_path, test_data_path):
    """ load train dataset and test dataset """
    sentence = Field(
        batch_first=True, 
        sequential=True, 
        tokenize="spacy",
    )
    gold_label = Field(
        batch_first=True, 
        sequential=False, 
        use_vocab=False, 
        dtype = torch.float)
    
    fields = [
        ('sentence_id', None),
        ('gold_label', gold_label),
        ('sentence', sentence)
    ]

    train_dataset = TabularDataset(path=train_data_path, format='csv', fields=fields, skip_header=True)
    print("train dataset: " + str(len(train_dataset.examples)))
    test_dataset = TabularDataset(path=test_data_path, format='csv', fields=fields, skip_header=True)
    print("test dataset: " + str(len(test_dataset.examples)))

    sentence.build_vocab(
        train_dataset, test_dataset,
        max_size=30000,
        vectors="glove.6B.300d",
        vectors_cache=".vector_cache",
        unk_init = torch.Tensor.normal_
    )

    return train_dataset, test_dataset, sentence


def k_fold_split(dataset, k, seed):
    """ split dataset into k folds """
    if k == 1:
        return [dataset]
    else:
        one_fold, other = dataset.split(1 / k, random_state=seed)
        results = k_fold_split(other, k - 1, seed)
        results.append(one_fold)
        return results


def binary_metrics(pred_labels, gold_labels):
    """ Returns metrics per batch """

    result = acc_precision_recall_f1(pred_labels, gold_labels)
    accuracy = result['acc']
    precision = result['precision']
    recall = result['recall']
    f1 = result['f1']

    return accuracy, precision, recall, f1


def train(model, iterators, optimizer, criterion):
    """ Train the model """
    model.train()
    
    epoch_loss = 0
    num_batches = 0
    for batch in itertools.chain(*[iter(iterator) for iterator in iterators]):
        num_batches += 1
        optimizer.zero_grad()
        predictions = model(batch.sentence).squeeze(1)
        loss = criterion(predictions, batch.gold_label)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / num_batches


def evaluate(model, iterator, criterion, eval_mode):
    """ Evaluate the model """
    model.eval()
    
    epoch_loss = 0
    total_pred_labels = None
    total_gold_labels = None
    for batch in iterator:
        with torch.no_grad():
            predictions = model(batch.sentence).squeeze(1)
            loss = criterion(predictions, batch.gold_label)
            epoch_loss += loss.item()

            pred_labels = torch.round(torch.sigmoid(predictions))
            if total_pred_labels is None:
                total_pred_labels = pred_labels
                total_gold_labels = batch.gold_label
            else:
                total_pred_labels = torch.cat((total_pred_labels, pred_labels), 0)
                total_gold_labels = torch.cat((total_gold_labels, batch.gold_label), 0)

    total_pred_labels = total_pred_labels.detach().cpu().numpy().astype(int)
    total_gold_labels = total_gold_labels.detach().cpu().numpy().astype(int)
    accuracy, precision, recall, f1 = binary_metrics(total_pred_labels, total_gold_labels)

    if eval_mode == "test":
        submission = pd.read_csv("data/SemEval-2020 Task5/sample_submission.csv")
        submission['pred_label'] = total_pred_labels.reshape(len(total_pred_labels), 1)
        output_pred_file = os.path.join("data/SemEval-2020 Task5/output", "cnn", "submission.csv")
        submission.to_csv(output_pred_file, index=False)

    return epoch_loss / len(iterator), accuracy, precision, recall, f1


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def print_cross_validation_results(valid_results_per_fold):
    """ print overall results (all folds) """
    print()
    print('overall results')
    print('%-9s %-9s %-9s %-9s %-9s' % ('fold', 'accuracy', 'precision', 'recall', 'f1'))
    avg_accuracy, avg_precision, avg_recall, avg_f1 = 0, 0, 0, 0
    for n_fold, results in enumerate(valid_results_per_fold):
        print('%-9d %-9.4f %-9.4f %-9.4f %-9.4f' % (n_fold + 1, results[0], results[1], results[2], results[3]))
        avg_accuracy += results[0]
        avg_precision += results[1]
        avg_recall += results[2]
        avg_f1 += results[3]
    avg_accuracy /= len(valid_results_per_fold)
    avg_precision /= len(valid_results_per_fold)
    avg_recall /= len(valid_results_per_fold)
    avg_f1 /= len(valid_results_per_fold)
    print('%-9s %-9.4f %-9.4f %-9.4f %-9.4f' % ('average', avg_accuracy, avg_precision, avg_recall, avg_f1))


def main(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set seed
    set_seed(args)

    # Load train dataset
    logger.info("Loading train dataset")
    train_dataset, test_dataset, sentence = load_dataset(args.data_path, args.test_data_path)

    # Initialize config
    config = TextCNNConfig()
    config.vocab_size = len(sentence.vocab)
    config.pretrained_embeddings = sentence.vocab.vectors
    config.unk_idx = sentence.vocab.stoi[sentence.unk_token]
    config.padding_idx = sentence.vocab.stoi[sentence.pad_token]

    # Split the train dataset into k folds for cross validation.
    train_splits = k_fold_split(train_dataset, args.num_folds, random.getstate())
    iterators = [
        BucketIterator(
            subset, 
            device=args.device, 
            batch_size=args.batch_size, 
            sort_key=lambda example: len(example.sentence), 
            sort_within_batch=True)
        for subset in train_splits
    ]

    # Train models in n_fold
    valid_results_per_fold = []
    for n_fold in range(len(iterators)):
        # split train-valid
        valid_iterator = iterators[n_fold]
        train_iterators = iterators[:n_fold] + iterators[n_fold + 1:]

        # Initialize a new model for each fold.
        model = CNN(config)

        # Set optimizer and criterion for training.
        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCEWithLogitsLoss()

        # Load the model and criterion.
        model.to(args.device)
        criterion.to(args.device)

        N_EPOCHS = 5
        best_valid_loss = float('inf')
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss = train(model, train_iterators, optimizer, criterion)
            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_iterator, criterion, "validation")
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'output/SemEval-2020-Task5/textcnn/model.pt')
    
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        # collect stats on validation dataset in the last epoch
        valid_results_per_fold.append((valid_acc, valid_precision, valid_recall, valid_f1))
    print_cross_validation_results(valid_results_per_fold)
    
    # Test
    test_iterator = BucketIterator(
        test_dataset, 
        device=args.device, 
        batch_size=args.batch_size, 
        sort_key=lambda example: len(example.sentence), 
        sort_within_batch=True)

    # Initialize a new model for test.
    model = CNN(config)

    # Set criterion for test.
    criterion = nn.BCEWithLogitsLoss()

    # Load the model and criterion.
    model.to(args.device)
    criterion.to(args.device)

    model.load_state_dict(torch.load('tut4-model.pt'))
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_iterator, criterion, "test")
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test Precision: {test_precision*100:.2f}% | Test Recall: {test_recall*100:.2f}% | Test F1: {test_f1*100:.2f}%')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default='data/SemEval-2020 Task5/csv/train.csv',
        type=str,
    )
    parser.add_argument(
        "--test_data_path",
        default='data/SemEval-2020 Task5/csv/test.csv',
        type=str,
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for train and validation")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=0, help="For distributed training: local_rank")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_folds", type=int, default=10, help="num of folds for cross validation")
    args = parser.parse_args()

    main(args)