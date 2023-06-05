"""
Description: This code implements a sentiment analysis classifier to predict the sentiment of a given sentence
Date: January 31, 2023
Author: Scott James Nelson
"""


import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn.metrics
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer



class BaselineClassifier():

    def __init__(self, args):
        """
        A baseline classifier that always predicts positive sentiment.

        Attributes
        --------------------
            train_sents -- list of training sentences as strings
            train_labels -- list of training integer (0 or 1) labels
            val_sents -- list of validation sentences as strings
            val_labels -- list of validation integer (0 or 1) labels

        """
        self.train_sents, self.train_labels = self.read_data(args.train_file)
        self.val_sents, self.val_labels = self.read_data(args.val_file)

    def read_data(self, filename):
        """
        Extracts all the sentences and labels from the input file.

        Parameters
        --------------------
            filename    -- filename as a string

        Returns
        --------------------
            sents       -- list of sentences as strings
            labels      -- list of integer (0 or 1) labels
        """
        sents = []
        labels = []
        with open(filename) as f:
            for line in f.readlines():
                line = line.strip().split(' ', 1) # only split once
                sents.append(line[1])
                labels.append(int(line[0]))
        return sents, labels

    def predict(self, corpus):
        """
        Always predicts a value of 1 given the input corpus.

        Parameters
        --------------------
            corpus    -- list of sentences

        Returns
        --------------------
            list of 1 for each sentence in the corpus
        """
        # part a
        return [1]* len(corpus)

    def evaluate(self):
        """
        Computes and prints accuracy on training and validation predictions.
        """
        # part a
        train_predictions = self.predict(self.train_sents)
        val_predictions = self.predict(self.val_sents)
        train_accuracy = self.compute_accuracy(self.train_labels, train_predictions)
        val_accuracy = self.compute_accuracy(self.val_labels, val_predictions)
        precision = sklearn.metrics.precision_score(self.val_labels, val_predictions)
        recall = sklearn.metrics.recall_score(self.val_labels, val_predictions)
        f1 = sklearn.metrics.f1_score(self.val_labels, val_predictions)
        macro_average = sklearn.metrics.f1_score(self.val_labels, val_predictions, average='macro')
        micro_average = sklearn.metrics.f1_score(self.val_labels, val_predictions, average='micro')
        print("Training accuracy: {:.4f}".format(train_accuracy))
        print("-"*round(train_accuracy*100))
        print("Validation accuracy: {:.4f}".format(val_accuracy))
        print("-"*round(val_accuracy*100))
        print("Precision: {:.4f}".format(precision))
        print("-"*round(precision*100))
        print("Recall: {:.4f}".format(recall))
        print("-"*round(recall*100))
        print("F1: {:.4f}".format(f1))
        print("-"*round(f1*100))
        print("Macro Average: {:.4f}".format(macro_average))
        print("Micro Average: {:.4f}".format(micro_average))
        print("Confusion Matrix: ")
        print(sklearn.metrics.confusion_matrix(self.val_labels, val_predictions))

    def compute_accuracy(self, labels, predictions):
        """
        Computes the accuracy given the labels, x, and predictions, y.

        Parameters
        --------------------
            labels      -- list of integer (0 or 1) labels
            predictions -- list of integer (0 or 1) predictions

        Returns
        --------------------
            accuracy    -- float between 0 and 1
        """
        # part a
        return sum(labels[i] == predictions[i] for i in range(len(labels))) / len(labels)


class NaiveBayesClassifier(BaselineClassifier):

    def __init__(self, args):
        """
        An sklearn Naive Bayes classifier with unigram features.

        Attributes
        --------------------
            train_sents -- list of training sentences as strings
            train_labels -- list of training integer (0 or 1) labels
            val_sents -- list of validation sentences as strings
            val_labels -- list of validation integer (0 or 1) labels
            vectorizer -- sklearn CountVectorizer for training data unigrams
            classifier -- sklearn MultinomialNB classifer object

        """
        super().__init__(args)
        # part b
        # TODO: Assign a CountVectorizer to self.vectorizer.
        self.vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1))
        #self.features = self.vectorizer.fit_transform(self.train_sents)
        #print(self.features)
        self.classifier = MultinomialNB()

        # TODO: Assign a new MultinomialNB() to self.classifier.
        self.train()

    def train(self):
        """
        Trains a Naive Bayes classifier on training sentences and labels.
        """
        # part b
        #  Compute features X from self.train_sents.
        trained_features_x = self.vectorizer.fit_transform(self.train_sents)

        #  Convert train_labels to a numpy array of labels y.
        trained_features_y = np.array(self.train_labels)

        #  Fit the classifier on X and y.
        self.classifier.fit(trained_features_x, trained_features_y)

    def predict(self, corpus):
        """
        Predicts labels on the corpus using the trained classifier.

        Parameters
        --------------------
            corpus    -- list of sentences

        Returns
        --------------------
            a list of predictions
        """
        # part b
        X = self.vectorizer.transform(corpus)
        return self.classifier.predict(X)


class LogisticRegressionClassifier(NaiveBayesClassifier):

    def __init__(self, args):
        """
        An sklearn Logistic Regression classifier with unigram features.

        Attributes
        --------------------
            train_sents -- list of training sentences as strings
            train_labels -- list of training integer (0 or 1) labels
            val_sents -- list of validation sentences as strings
            val_labels -- list of validation integer (0 or 1) labels
            vectorizer -- sklearn CountVectorizer for training data unigrams
            classifier -- sklearn LogisticRegression classifer object

        """
        BaselineClassifier.__init__(self, args)
        # part c
        
        # TODO: Assign a CountVectorizer to self.vectorizer.
        self.vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1))

        # TODO: Assign a new LogisticRegression() to self.classifier.
        self.classifier = LogisticRegression(penalty=args.penalty, C=args.C, solver='liblinear')
        # Hint: You can adjust penalty and C params with command-line args.
        
        #self.features = self.vectorizer.fit_transform(self.train_sents)
        self.train()


class BigramLogisticRegressionClassifier(LogisticRegressionClassifier):

    def __init__(self, args):
        """
        A Logistic Regression classifier with unigram and bigram features.

        Attributes
        --------------------
            train_sents -- list of training sentences as strings
            train_labels -- list of training integer (0 or 1) labels
            val_sents -- list of validation sentences as strings
            val_labels -- list of validation integer (0 or 1) labels
            vectorizer -- sklearn CountVectorizer for unigrams and bigrams
            classifier -- sklearn LogisticRegression classifer object

        """
        BaselineClassifier.__init__(self, args)
        # part d
        # TODO: Assign a CountVectorizer to self.vectorizer.
        # TODO: Assign a new LogisticRegression() to self.classifier.
        # Hint: Be sure to set args.solver.
        self.vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
        self.classifier = LogisticRegression(solver=args.solver)
        self.train()


def main(args):
    # part a: Evaluate basline classifier (i.e., always predicts positive).
    # Hint: Should see roughly 50% accuracy.
    print("\tBaseline classifier results:")
    BaselineClassifier(args).evaluate()
    print('='*100)

    # part b: Evaluate Naive Bayes classifier with unigram features.
    # Hint: Should see over 90% training and 70% testing accuracy.
    print("\n\tNaive Bayes classifier results:")
    NaiveBayesClassifier(args).evaluate()
    print('='*100)

    # part c: Evaluate logistic regression classifier with unigrams.
    # Hint: Should see over 95% training and 70% testing accuracy.
    print("\n\tLogistic regression classifier results:")
    LogisticRegressionClassifier(args).evaluate()
    print('='*100)

    # part d: Evaluate logistic regression classifier with unigrams + bigrams.
    # Hint: Should see over 95% training and 70% testing accuracy.
    print("\n\tBigram logistic regression classifier results:")
    BigramLogisticRegressionClassifier(args).evaluate()
    print('='*100)

    pass

def evaluate(self):
    """
    Evaluates the classifier on the training and validation data.
    """
    # part a


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='sentiment-data/train.txt')
    parser.add_argument('--val_file', default='sentiment-data/val.txt')
    parser.add_argument('--solver', default='liblinear', help='Optimization algorithm.')
    parser.add_argument('--penalty', default='l2', help='Regularization for logistic regression.')
    parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength for logistic regression.')

    args = parser.parse_args()
    main(args)
