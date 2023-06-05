import unittest
from io import StringIO
import sys

# Assuming the original code is in a script called bigram_language_model.py
from lm import LanguageModel


class TestLanguageModel(unittest.TestCase):
    def setUp(self):
        """
        This method is called before each test. We're using it to set up the parameters for the language model.
        """
        class Args:
            alpha = 0.0001
            train_file = 'test_train.txt'
            val_file = 'test_val.txt'
            train_fraction = 1.0
            show_plot = False

        self.args = Args()
        self.languageModel = LanguageModel(self.args)

    def test_tokenize(self):
        """
        Test the tokenize method with a simple sentence.
        """
        with open('test_tokenize.txt', 'w') as f:
            f.write("This is a test sentence.")
        tokens = self.languageModel.tokenize('test_tokenize.txt')
        self.assertEqual(tokens, ['this', 'is', 'a', 'test', 'sentence', '.'])

    def test_make_vocab(self):
        """
        Test the make_vocab method with some tokens.
        """
        tokens = ['this', 'is', 'a', 'test', 'sentence', '.', 'this', 'is', 'another', 'test', 'sentence', '.']
        vocab = self.languageModel.make_vocab(tokens)
        self.assertEqual(vocab, {'this': 2, 'is': 2, 'a': 1, 'test': 2, 'sentence': 2, '.': 2, 'another': 1})

    def test_compute_perplexity(self):
        """
        Test the compute_perplexity method with a simple case.
        """
        tokens = ['this', 'is', 'a', 'test']
        self.languageModel.vocab = {'this': 1, 'is': 1, 'a': 1, 'test': 1}
        self.languageModel.token_to_idx = {'this': 0, 'is': 1, 'a': 2, 'test': 3}
        self.languageModel.bigrams = np.ones((4, 4)) / 4
        perplexity = self.languageModel.compute_perplexity(tokens)
        self.assertAlmostEqual(perplexity, 4.0)  # The perplexity should be equal to the size of the vocabulary


if __name__ == '__main__':
    unittest.main()
