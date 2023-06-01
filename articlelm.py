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
import sys
import pandas as pd
import argparse
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Define the file path manually
csv_file_path = r'C:\Users\user1\Desktop\AI\SURP\nlp\data\unique_articles_df.csv'

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

# Define functions
def plot_ngrams(corpus, n, N):
    """                 N-grams Frequency Plot: 
    This function can take in a corpus and a parameter for 'n', 
    and it will generate a frequency plot of the top N most common n-grams. 
    This can give a visual sense of the common phrases in the corpus.
    """
    ngrams = list(nltk.ngrams(corpus.split(), n))
    ngrams_frequencies = Counter(ngrams)
    most_common_ngrams = ngrams_frequencies.most_common(N)

    plt.figure(figsize=(10, 5))
    ngrams, frequencies = zip(*most_common_ngrams)
    ngrams = [' '.join(gram) for gram in ngrams]
    plt.bar(ngrams, frequencies)
    plt.show()

def vectorize_corpus(corpus):
    """ Term Frequency-Inverse Document Frequency (TF-IDF) Vectorization: 
    This function can be used to convert a corpus of text documents into vectors 
    that reflect the importance of each term to each document in the corpus.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

def generate_wordcloud(text):
    """                     Word Cloud: 
    Word clouds can be a fun and visually appealing way to get a sense of 
    the most frequent words in a corpus.
    """
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = None, 
                min_font_size = 10).generate(text)

    plt.figure(figsize = (4, 4), facecolor = None) 
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

def sentiment_analysis(text):
    """                 Sentiment Analysis: 
    This function uses a pre-trained sentiment analysis model to estimate the sentiment of text. 
    It can be useful for understanding overall positive or negative sentiments in a corpus.
    """
    sia = SentimentIntensityAnalyzer()
    polarity = sia.polarity_scores(text)
    return polarity

def pos_tagging(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags

def find_words_with_special_chars(text):
    """The pattern \b\w*\W\w*\b matches any word that contains at least one non-word character.
    """
    pattern = r'\b\w*\W\w*\b'
    matches = re.findall(pattern, text)
    return matches

def find_all_caps(text):
    """The pattern \b\w*\W\w*\b matches any word that contains at least one non-word character.
    """
    matches = re.findall(r'\b[A-Z\s]+\b', text)
    return matches

def find_awk(text):
    matches = re.findall(r'â€', text)
    return matches

def find_alphaNumeric(text):
    matches = re.findall(r'[^a-zA-Z0-9\s]+', text)
    return matches

def regex(corpus):

    # use regular expressions to find repeated words
    # repeated_words = find_words_with_special_chars(corpus)
    # # print("Repeated words:", repeated_words)

    # all_caps_words = find_all_caps(corpus)
    # print("ALL CAPS words:", set(all_caps_words))

    # # use word tokenization to split the text into words
    # tokens = nltk.word_tokenize(corpus)

    # # normalize the tokens
    # normalized_tokens = [token.lower() for token in tokens if token.isalpha()]

    # # apply stemming
    # stemmer = PorterStemmer()
    # stemmed_tokens = [stemmer.stem(token) for token in normalized_tokens]

    # print("Normalized and stemmed tokens:", stemmed_tokens)
    # print(find_awk(corpus))
    print(f"\nset of non-alphanumeric: {set(find_alphaNumeric(corpus))}")
    for na in set(find_alphaNumeric(corpus)):
        print('\n',na)
    print('%'*60)

def find_sequence_in_sentences(corpus, sequence):
    # split the corpus into sentences
    occurances = []
    sentences = corpus.split('.')

    # iterate over the sentences
    for sentence in sentences:
        # if the sequence is found in the sentence
        if re.search(sequence, sentence):
            # return the sentence
            occurances.append(sentence)
    if len(occurances) == 0:
        return None
    return occurances

    # return None if the sequence was not found in any sentence
    return None







# Main script logic
def main(args):
    languageModel = bigramLanguageModel(args)
    print(f"\n\n\n {args.val_file} \n\n\t{args.train_file}")
    # languageModel.plot_vocab(languageModel.vocab)


    #vectorize_corpus(corpus):
    # tfidf_matrix, vectorizer = vectorize_corpus(df['content'])
    # print(tfidf_matrix, vectorizer)
    all_strings_content = " ".join(list(df['content']))
    # for na in set(find_alphaNumeric(all_strings_content)):
    #     print(na)
    #     sequence = na
    # print('\n', find_sequence_in_sentences(all_strings_content,r'â€'))
    # print(len(all_strings_content))
    generate_wordcloud(all_strings_content)
    regex(all_strings_content)

    sentiment = sentiment_analysis(all_strings_content)
    print(f"sentiment: \t{sentiment}")

    # pos = pos_tagging(args.train_file)
    # print(f"\n\n\n pos tags: \t {pos}")


    # plot_ngrams(corpus, n, N):
    plot_ngrams(all_strings_content, 5, 2)

"""
FELONEOUS BEHAVIOR:
    CLICK HERE TO GET THE FOX NEWS APP
    Get all the stories you need-to-know from the most powerful name in news delivered first thing every morning to your inbox Subscribed You've successfully subscribed to this newsletter!
    Fox News Flash top headlines are here. Check out what's clicking on Foxnews.com.
    Sign up for CNN Opinion’s newsletter.Join us on Twitter and Facebook

    â€™ --> '
    â€˜
    â€¦
    \' ( in Here\'s )
"""





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

    # Process the DataFrame as needed
    print(df.head())
    main(args)
