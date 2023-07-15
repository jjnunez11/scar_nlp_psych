import time
import pandas as pd
from utils import print_from_history, series_to_matrix
from copy import deepcopy
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
import numpy as np
from evaluators.calculate_metrics import add_epoch_perf
import os
import _pickle as cPickle
import bz2


class RuleTrainer(object):
    METRICS = ['acc', 'bal_acc', 'auc', 'prec', 'rec', 'f1', 'spec', 'loss']  # Performance metrics to evaluate

    def __init__(self,
                 rule_token_idx):

        self.rule_token_idx = rule_token_idx
        self.best_dev_f1 = 0
        self.start = None

    def fit(self,
            train_data,
            dev_data,
            test_data,
            n_epochs: int = 100):

        train_history = pd.DataFrame()
        dev_history = pd.DataFrame()
        test_history = pd.DataFrame()

        # Set Pandas setting to allow more columns to print out epoch results
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 100)

        # Load the data
        train_x = series_to_matrix(deepcopy(train_data['vector']))
        train_y = deepcopy(train_data['label']).to_numpy()
        dev_x = series_to_matrix(deepcopy(dev_data['vector']))
        dev_y = deepcopy(dev_data['label']).to_numpy()
        test_x = series_to_matrix(deepcopy(test_data['vector']))
        test_y = deepcopy(test_data['label']).to_numpy()

        self.start = time.time()

        warnings.warn('No training is actually done for the rule method, just evaluates rule')

        for epoch in range(n_epochs):
            # Predict using train set
            print('here are the type and shape of train_x:')
            print(type(train_x))
            print(train_x.shape)

            predicted_labels = np.where(train_x[:, self.rule_token_idx] > 0, 1.000001, 0)
            target_labels = np.array(train_y)

            train_loss = 0  # np.mean(train_loss)

            # Evaluate performance on training set and update
            train_history = add_epoch_perf(target_labels, predicted_labels, train_loss, train_history)

            # Evaluate on dev set ----------------------------------------
            predicted_labels = np.where(dev_x[:, self.rule_token_idx] > 0, 1.000001, 0)
            target_labels = np.array(dev_y)
            dev_loss = 0  # np.mean(dev_loss)
            dev_history = add_epoch_perf(target_labels, predicted_labels, dev_loss, dev_history)

            # Evaluate on test set ----------------------------------------
            predicted_labels = np.where(test_x[:, self.rule_token_idx] > 0, 1.000001, 0)
            target_labels = np.array(test_y)
            test_loss = 0
            test_history = add_epoch_perf(target_labels, predicted_labels, test_loss, test_history)

            # Print Epoch Results so far
            print_from_history(dev_history, -1, self.start, epoch, n_epochs)

        return train_history, dev_history, test_history, self.start

    def eval_only(self, test_data):

        raise NotImplementedError("Have not implemented eval_only for rule-based classifier as it is not relevant")
