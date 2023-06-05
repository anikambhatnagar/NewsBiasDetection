"""
Description: This code is an implementation of a bigram language model with add-alpha smoothing in Python
Date: January 2023
Author: Scott James Nelson ^& William Judy (collaborator)
"""

#import this
import math
import nltk
import argparse
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.tokenize import word_tokenize




class LanguageModel:

    def __init__(self, args):
        """
        A bigram language model with add-alpha smoothing.

        Attributes
        --------------------
            alpha -- alpha for add-alpha smoothing
            train_tokens -- list of training data tokens
            val_tokens -- list of validation data tokens
            vocab -- vocabulary frequency dict (key = word, val = frequency)
            token_to_idx -- vocabulary index dict (key = word, val = index)
            bigrams -- 2D np array of bigram probabilities, where each (i, j)
                       value is the smoothed prob of the bigram starting with
                       vocab token index i followed by vocab token index j
        """
        self.alpha = args.alpha
        self.train_tokens = self.tokenize(args.train_file)
        self.val_tokens = self.tokenize(args.val_file)

        # Use only the specified fraction of training data.
        num_samples = int(args.train_fraction * len(self.train_tokens))
        self.train_tokens = self.train_tokens[: num_samples]
        self.vocab = self.make_vocab(self.train_tokens)
        self.token_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.bigrams = self.compute_bigrams(self.train_tokens)

    def get_indices(self, tokens):
        """
        Converts each of the string tokens to indices in the vocab.

        Parameters
        --------------------
            tokens    -- list of tokens

        Returns
        --------------------
            list of token indices in the vocabulary
        """
        # part b
        return [self.token_to_idx[token] for token in tokens if token in self.token_to_idx]

    def compute_bigrams(self, tokens):
        """
        Populates probability values for a 2D np array of all bigrams.

        Parameters
        --------------------
            tokens    -- list of tokens
            alpha     -- alpha for add-alpha smoothing

        Returns
        --------------------
            bigrams   -- 2D np array of bigram probabilities, where each (i, j)
                       value is the smoothed prob of the bigram starting with
                       vocab token index i followed by vocab token index j
        """
        counts = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
        probs = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
        tokens = self.get_indices(tokens)

        # part c
        for i in range(len(tokens)-1):
            counts[tokens[i]][tokens[i+1]] +=1
        for vocab_i in tqdm(range(len(self.vocab))):
            probs[vocab_i, :] = (counts[vocab_i] + self.alpha) / (sum(counts[vocab_i]) + self.alpha*(len(self.vocab)))
        return probs

    def compute_perplexity(self, tokens):
        """
        Evaluates the LM by calculating perplexity on the given tokens.

        Parameters
        --------------------
            bigrams    -- 2D np array of bigram probabilities, where each (i, j)
                       value is the smoothed prob of the bigram starting with
                       vocab token index i followed by vocab token index j
            tokens     -- list of tokens

        Returns
        --------------------
            perplexity
        """
        tokens = self.get_indices(tokens)

        # part c
        ll=0
        for eaToken in range(len(tokens)-1):
            ll += math.log2(self.bigrams[ tokens[eaToken] ][ tokens[eaToken+1] ])
        return 2**(-ll/len(tokens))

    def tokenize(self, corpus):
        """
        Splits the given corpus file into tokens using nltk's tokenizer.

        Parameters
        --------------------
            corpus    -- filename as a string

        Returns
        --------------------
            tokens    -- list of tokens
        """
        return word_tokenize(open(corpus, 'r').read().lower())

    def make_vocab(self, train_tokens):
        """
        Creates a vocabulary dictionary that maps tokens to frequencies.

        Parameters
        --------------------
            train_tokens    -- list of training tokens

        Returns
        --------------------
            vocab           -- vocab frequency dict (key = word, val = freq)
        """
        # part b
        unique_words = list(set(train_tokens))
        vocab = {}
        for i in range(len(unique_words)):
            vocab[unique_words[i]] = train_tokens.count(unique_words[i])
        return vocab

    def plot_vocab(self, vocab):
        """
        Plots words from most to least common, with frequency on the y-axis.

        Parameters
        --------------------
            vocab           -- vocab frequency dict (key = word, val = freq)
        """
        # part b
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        x = [i for i in range(len(sorted_vocab))]
        y = [sorted_vocab[i][1] for i in range(len(sorted_vocab))]
        plt.plot(x, y, label='normal')
        plt.xlabel('Word Rank')
        plt.ylabel('Frequency')
        plt.title('Word Frequencies')
        plt.legend()
        plt.show()



def main(args):
    languageModel = LanguageModel(args)

    # part b: Plot word frequencies by setting command-line arg show_plot.
    if args.show_plot:
        languageModel.plot_vocab(languageModel.vocab)

    # part c: Plot training and validation perplexities as a function of alpha.
    # Hint: Expect ~136 for train and 530 for val when alpha=0.017
    plot_alphas()

    # part d: Plot train/val perplexities for varying amounta of training data.
    plot_train_fractions()

def plot_alphas():
    alpha_levels = [10**i for i in range(-5,2,1)]
    print(alpha_levels)
    train_ppl=[]
    val_ppl=[]
    lm_dict = {alpha : LanguageModel(customParam(alpha, 1.0)) for alpha in alpha_levels}
    for a in tqdm(alpha_levels):
        lm_a = lm_dict.get(a)
        train_ppl.append(lm_a.compute_perplexity(lm_a.train_tokens))
        val_ppl.append(lm_a.compute_perplexity(lm_a.val_tokens))
    x = range(-5, 2)
    plt.plot(x, train_ppl, label='Training')
    plt.plot(x, val_ppl, label='Validation')
    plt.title('Perplexity with different smoothing Alphas')
    plt.xlabel('Alpha')
    plt.ylabel('Perplexity')
    plt.xticks(x, alpha_levels)
    plt.yticks(np.linspace(0, max(val_ppl), 5))
    plt.legend()
    plt.show()

def plot_train_fractions():
    train_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    train_ppl=[]
    val_ppl=[]
    lm_dict = {train_fraction : LanguageModel(customParam(0.001, train_fraction)) for train_fraction in train_fractions}
    for tf in tqdm(train_fractions):
        lm_tf=lm_dict.get(tf)
        train_ppl.append(lm_tf.compute_perplexity(lm_tf.train_tokens))
        val_ppl.append(lm_tf.compute_perplexity(lm_tf.val_tokens))
    x = range(1, 11)
    plt.plot(x, train_ppl, label='training')
    plt.plot(x, val_ppl, label='validation')
    plt.title('Perplexity w/ varying percentage of Training Data')
    plt.xlabel('Training Data')
    plt.ylabel('Perplexity')
    plt.xticks(x, train_fractions)
    plt.yticks(np.linspace(0, max(val_ppl), 7))
    plt.legend()
    plt.show()

def customParam(alpha, train_fraction):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='lm-data/brown-train.txt')
    parser.add_argument('--val_file', default='lm-data/brown-val.txt')
    parser.add_argument('--train_fraction', type=float, default=train_fraction, help='Specify a fraction of training data to use to train the language model.')
    parser.add_argument('--alpha', type=float, default=alpha, help='Parameter for add-alpha smoothing.')
    parser.add_argument('--show_plot', type=bool, default=True, help='Whether to display the word frequency plot.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    df = pd.read_csv("../data/unique_articles_df.csv")
    print(df[0])
    parser.add_argument('--train_file', default='lm-data/brown-train.txt')
    parser.add_argument('--val_file', default='lm-data/brown-val.txt')
    parser.add_argument('--train_fraction', type=float, default=1.0, help='Specify a fraction of training data to use to train the language model.')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Parameter for add-alpha smoothing.')
    parser.add_argument('--show_plot', type=bool, default=False, help='Whether to display the word frequency plot.')

    args = parser.parse_args()
    main(args)
