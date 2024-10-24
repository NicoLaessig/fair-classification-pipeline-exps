from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from .utils import performance_func, measure_error
except:
    from utils import performance_func, measure_error

class Model:
        def __init__(self, model_type, x_train, y_train, edge_train, w_edge_train, x_test, y_test, edge_test, w_edge_test, x_val, y_val):
            """         
                Args: 
                    model_type: Choose a model type among LogisticRegression, RandomForest, and NeuralNetwork
                    x_train: Features for training data
                    y_train: Labels for training data
                    w_train: Similarity matrix for training data
                    edge_train: Indices of similar pairs for training data
                    w_edge_train: Similarity values of similar pairs for training data
                    x_test: Features for test data
                    y_test: Labels for test data
                    w_test: Similarity matrix for test data
                    edge_test: Indices of similar pairs for test data
                    w_edge_train: Similarity values of similar pairs for test data
                    x_val: Features for validation data
                    y_val: Labels for validation data
            """

            self.model_type = model_type
            
            self.x_train = x_train
            self.y_train = y_train
            self.edge_train = edge_train
            self.w_edge_train = w_edge_train

            self.x_test = x_test
            self.y_test = y_test
            self.edge_test = edge_test
            self.w_edge_test = w_edge_test
            
            self.x_val = x_val
            self.y_val = y_val

            self.random_state_arr = [0, 1, 2, 3, 4]
            
        def train(self):
            """         
                Trains the selected model on training data and return both trainining and test performance.

                Return:
                    train_performance: Test accuracy and consistency score for training data
                    test_performance: Test accuracy and consistency score for test data
            """

            if len(np.unique(self.y_train)) < 2:
                y_hat = copy.deepcopy(self.y_test)
                y_hat[:] = self.y_train[0]

                test_accuracy = np.sum(y_hat == self.y_test) / len(y_hat)
     
                total_error = measure_error(y_hat, self.edge_test, self.w_edge_test)
                num_similar_pairs = len(self.edge_test)
                test_consistency_score = 1 - total_error / num_similar_pairs

                train_performance = [1.0, 1.0]
                test_performance = [test_accuracy, test_consistency_score]

            elif self.model_type == "LogisticRegression": 
                C_arr = [0.1, 1, 10, 100]
                clf, train_performance, test_performance = self.train_lr(C_arr)
                
            elif self.model_type == "RandomForest":
                n_estimators_arr = [50, 100, 200]
                min_samples_split_arr = [2, 10]
                max_depth_arr = [5, None]

                clf, train_performance, test_performance = self.train_rf(n_estimators_arr, min_samples_split_arr, max_depth_arr)
                
            elif self.model_type == "NeuralNetwork":
                learning_rate_arr = [0.1, 0.01, 0.001, 0.0001]
                clf, train_performance, test_performance = self.train_nn(learning_rate_arr)

            pred = clf.predict(self.x_test)
                
            return pred, train_performance, test_performance
        
        def train_lr(self, C_arr):
            """         
                Trains the selected model on training data and return both trainining and test performance.

                Args:
                    C_arr: Regularization strength in logistic regression

                Return:
                    train_performance: Test accuracy and consistency score for training data
                    test_performance: Test accuracy and consistency score for test data
            """

            val_acc = []
            for C in C_arr:
                clf = LogisticRegression(solver='lbfgs', C=C, random_state=0)
                _ = clf.fit(self.x_train, self.y_train)    
                val_acc.append(clf.score(self.x_val, self.y_val))
            best_idx = np.argmax(val_acc)

            train_acc, train_cs, test_acc, test_cs, = 0, 0, 0, 0
            for random_state in self.random_state_arr:
                clf = LogisticRegression(solver='lbfgs', C=C_arr[best_idx], random_state=random_state)
                clf.fit(self.x_train, self.y_train)

                train_result = performance_func(clf, self.x_train, self.y_train, self.edge_train, self.w_edge_train)
                train_acc += train_result[0] / len(self.random_state_arr)
                train_cs += train_result[1] / len(self.random_state_arr)

                test_result = performance_func(clf, self.x_test, self.y_test, self.edge_test, self.w_edge_test)
                test_acc += test_result[0] / len(self.random_state_arr)
                test_cs += test_result[1] / len(self.random_state_arr)

            return clf, (train_acc, train_cs), (test_acc, test_cs)
    
        def train_rf(self, n_estimators_arr, min_samples_split_arr, max_depth_arr):
            """         
                Trains the selected model on training data and return both trainining and test performance.

                Args:
                    n_estimators_arr: Number of trees in the forest
                    min_samples_split_arr: The minimum number of samples required to split
                    max_depth_arr: The maximum depth of the tree

                Return:
                    train_performance: Test accuracy and consistency score for training data
                    test_performance: Test accuracy and consistency score for test data
            """

            val_acc = []
            for n_estimators in n_estimators_arr:
                for min_samples_split in min_samples_split_arr:
                    for max_depth in max_depth_arr:
                        clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split = min_samples_split, max_depth=max_depth, random_state=0)
                        _ = clf.fit(self.x_train, self.y_train)    
                        val_acc.append(clf.score(self.x_val, self.y_val))
            best_idx = np.argmax(val_acc)

            train_acc, train_cs, test_acc, test_cs, = 0, 0, 0, 0
            for random_state in self.random_state_arr:
                clf = RandomForestClassifier(n_estimators=n_estimators_arr[(best_idx//(len(min_samples_split_arr)*len(max_depth_arr)))%len(n_estimators_arr)], min_samples_split = min_samples_split_arr[(best_idx//len(max_depth_arr))%len(min_samples_split_arr)], max_depth=max_depth_arr[best_idx % len(max_depth_arr)], random_state=random_state)
                clf.fit(self.x_train, self.y_train)

                train_result = performance_func(clf, self.x_train, self.y_train, self.edge_train, self.w_edge_train)
                train_acc += train_result[0] / len(self.random_state_arr)
                train_cs += train_result[1] / len(self.random_state_arr)

                test_result = performance_func(clf, self.x_test, self.y_test, self.edge_test, self.w_edge_test)
                test_acc += test_result[0] / len(self.random_state_arr)
                test_cs += test_result[1] / len(self.random_state_arr)
    
            return clf, (train_acc, train_cs), (test_acc, test_cs)
        
        def train_nn(self, learning_rate_arr):
            """         
                Trains the selected model on training data and return both trainining and test performance.

                Args:
                    learning_rate_arr: The initial learning rate

                Return:
                    train_performance: Test accuracy and consistency score for training data
                    test_performance: Test accuracy and consistency score for test data
            """

            val_acc = []
            for learning_rate in learning_rate_arr:
                clf = MLPClassifier(learning_rate_init=learning_rate, hidden_layer_sizes=(10), alpha=0.1, random_state=0)
                _ = clf.fit(self.x_train, self.y_train)
                val_acc.append(clf.score(self.x_val, self.y_val))
            best_idx = np.argmax(val_acc)

            train_acc, train_cs, test_acc, test_cs, = 0, 0, 0, 0
            for random_state in self.random_state_arr:
                clf = MLPClassifier(learning_rate_init=learning_rate_arr[best_idx], hidden_layer_sizes=(10), alpha=0.1, random_state=random_state)
                clf.fit(self.x_train, self.y_train)

                train_result = performance_func(clf, self.x_train, self.y_train, self.edge_train, self.w_edge_train)
                train_acc += train_result[0] / len(self.random_state_arr)
                train_cs += train_result[1] / len(self.random_state_arr)

                test_result = performance_func(clf, self.x_test, self.y_test, self.edge_test, self.w_edge_test)
                test_acc += test_result[0] / len(self.random_state_arr)
                test_cs += test_result[1] / len(self.random_state_arr)
    
            return clf, (train_acc, train_cs), (test_acc, test_cs)

        