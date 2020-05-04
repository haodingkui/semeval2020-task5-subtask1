import sys
sys.path.append("utils")
from processors import Subtask1Processor

import unittest
import argparse

class TestCrossValidation(unittest.TestCase):

    def test_cv_data(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()

        args.data_dir = "/users5/dkhao/github/nlpexperiments-text-classification/data/SemEval-2020-Task5/"
        args.fold_number = 0
        args.num_folds = 10

        processor = Subtask1Processor(args)

        examples = processor.examples
        train_examples = processor.get_train_fold_examples()
        dev_examples = processor.get_dev_fold_examples()

        print(dev_examples[0].guid)

        # self.assertEqual(len(train_examples), len(examples) / num_folds * (num_folds - 1))
        # self.assertEqual(len(dev_examples), len(examples) / num_folds * 1)

    def test_cv_logic(self):
        training_examples = [x for x in range(100)]

        fold_size = int(len(training_examples) / 10)

        def get_dev_and_training_for_fold(fold):
            dev_set = training_examples[fold_number * fold_size : (fold_number + 1) * fold_size]
            train_set = training_examples[0:fold_number * fold_size] + training_examples[(fold_number + 1) * fold_size : ]
            return dev_set, train_set

        fold_number = 0
        dev_set, train_set = get_dev_and_training_for_fold(fold_number)

        self.assertEqual(dev_set, [x for x in range(10)])
        self.assertEqual(train_set, [x for x in range(10, 100)])

        fold_number = 5
        dev_set, train_set = get_dev_and_training_for_fold(fold_number)

        expected = [x for x in range(50, 60)]
        self.assertEqual(dev_set, expected)
        expected = [x for x in range(50)] + [x for x in range(60, 100)]
        self.assertEqual(train_set, expected)

if __name__ == "__main__":
    unittest.main()