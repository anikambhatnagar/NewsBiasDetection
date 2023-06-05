"""
Module Description: This script will evaluate the article content with a bigram language model.
"Scott Nelson"
"""

import math
import random
import nltk
import re
from nltk.stem import PorterStemmer
from collections import Counter
import tqdm
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
import sys
import argparse
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Define the file path manually
csv_file_path = r'C:\Users\user1\Desktop\AI\SURP\nlp\data\content_df.csv'

class bigramLanguageModel:
    def __init__(self, args):
        self.alpha = args.alpha
        self.train_tokens = self.tokenize(args.train_file)
        self.val_tokens = self.tokenize(args.val_file)

        num_samples = int(args.train_fraction * len(self.train_tokens))
        self.train_tokens = self.train_tokens[: num_samples]
        self.vocab = self.make_vocab(self.train_tokens)
        self.token_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.bigrams = self.compute_bigrams(self.train_tokens)

    def tokenize(self, corpus):
        return word_tokenize(corpus.lower())
    def compute_bigrams(self, tokens):
        counts = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
        probs = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
        tokens = self.get_indices(tokens)
        for i in range(len(tokens)-1):
            counts[tokens[i]][tokens[i+1]] +=1
        for vocab_i in range(len(self.vocab)):
            probs[vocab_i, :] = (counts[vocab_i] + self.alpha) / (sum(counts[vocab_i]) + self.alpha*(len(self.vocab)))
        return probs
    def make_vocab(self, tokens):
        unique_words = list(set(tokens))
        vocab = {}
        for i in range(len(unique_words)):
            vocab[unique_words[i]] = tokens.count(unique_words[i])
        return vocab
    def get_indices(self, tokens):
        return [self.token_to_idx[token] for token in tokens if token in self.token_to_idx]
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

'''
spam_filter.py
Spam v. Ham Classifier trained and deployable upon short
phone text messages.
'''

class SpamFilter:

    def __init__(self, text_train, labels_train):
        """
        Creates a new text-message SpamFilter trained on the given text 
        messages and their associated labels. Performs any necessary
        preprocessing before training the SpamFilter's Naive Bayes Classifier.
        As part of this process, trains and stores the CountVectorizer used
        in the feature extraction process.
        
        :param DataFrame text_train: Pandas DataFrame consisting of the
        sample rows of text messages
        :param DataFrame labels_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each text message
        """

        self.vectorizer = CountVectorizer(stop_words='english')
        self.features = self.vectorizer.fit_transform(text_train)
        self.classifier = MultinomialNB()
        self.classifier.fit(self.features, labels_train)
        return
        
    def classify (self, text_test):
        """
        Takes as input a list of raw text-messages, uses the SpamFilter's
        vectorizer to convert these into the known bag of words, and then
        returns a list of classifications, one for each input text
        
        :param list/DataFrame text_test: A list of text-messages (strings) consisting
        of the messages the SpamFilter must classify as spam or ham
        :return: A list of classifications, one for each input text message
        where index in the output classes corresponds to index of the input text.
        """
        
        test_features = self.vectorizer.transform(text_test)
        classifications = list(self.classifier.predict(test_features))
        return classifications
    
    def test_model (self, text_test, labels_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test texts
        and their associated labels), classifies each text, and then prints
        the classification_report on the expected vs. given labels.
        
        :param DataFrame text_test: Pandas DataFrame consisting of the
        test rows of text messages
        :param DataFrame labels_test: Pandas DataFrame consisting of the
        test rows of labels pertaining to each text message
        """

        y_actual = self.classify(text_test)
        return classification_report(labels_test, y_actual)
    
        
def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with only the message
    texts and labels as the remaining columns.
    
    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the texts
    and labels
    """
    
    df = pd.read_csv(data_file, encoding='latin-1')
    df = df.dropna(axis='columns')
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    return df

    

# Main script logic
def main(args):
    languageModel = bigramLanguageModel(args)
    print(f"\n\n\n {args.val_file} \n\n\t{args.train_file}")

    all_strings_content = " ".join(list(df['content']))






if __name__ == "__main__":
    # Code to be executed when the script is run directly
    # Call functions, perform operations, etc.
    # Read the CSV file into a Pandas DataFrame
    print('\n\n\n', '%'*450)
    parser = argparse.ArgumentParser()
    df = pd.read_csv(csv_file_path)
    # print(f"\n\n\t {df['title'][0]} \n\t{df['content'][0]}")
    random_example_int = random.randint(0, df.shape[0])
    print(random_example_int)

    parser.add_argument('--train_file', default=df['content'][random_example_int])
    parser.add_argument('--val_file', default=df['title'][random_example_int])
    parser.add_argument('--train_fraction', type=float, default=1.0, help='Specify a fraction of training data to use to train the language model.')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Parameter for add-alpha smoothing.')
    parser.add_argument('--show_plot', type=bool, default=False, help='Whether to display the word frequency plot.')
    args = parser.parse_args()

    print("\n\n> Head < ")
    print(df.head())
    print("\n\n> DESCRIBE < ")
    print(df.describe())
    print("\n\n\n")

    ###
    y_train, y_test, x_train, x_test = train_test_split(df['label'], df['text'], test_size=0.2)
    s = SpamFilter(x_train, y_train)
    s.test_model(x_test, y_test)
    print("\n\n  > TEST MODEL < ")
    print(s.test_model(x_test, y_test))
    print("\n\n  > CLASSIFY < ")
    spamCount = 0
    hamCount = 0
    for i in range(len(s.classify(x_test))):
        if s.classify(x_test)[i] == 'spam':
            spamCount += 1
        else:
            hamCount += 1
        #print(s.classify(x_test)[i], x_test.iloc[i])
    print("Spam Count: ", spamCount)
    print("Ham Count: ", hamCount)
    #print ham to spam ratio
    print("Ham to Spam Ratio: ", hamCount/spamCount)
    # Process the DataFrame as needed
    print(df.head())
    main(args)