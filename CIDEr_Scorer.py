"""# CIDEr Score"""

import math
from collections import defaultdict


class CiderScorer:
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.cache = defaultdict(lambda: None)

    def word_frequency(self, sentence):
        """
        This function takes a sentence as input and returns a dictionary of word frequencies.
        """
        if self.cache[sentence] is not None:
            return self.cache[sentence]

        freq = defaultdict(int)
        for word in sentence.split():
            freq[word] += 1

        self.cache[sentence] = freq
        return freq

    def calculate_cider_score(self, sentence1, sentence2):
        """
        This function takes two sentences as input and returns their CIDEr score.
        """
        # Get word frequencies for both sentences
        freq1 = self.word_frequency(sentence1)
        freq2 = self.word_frequency(sentence2)

        # Get the set of unique words in both sentences
        unique_words = set(freq1.keys()).union(set(freq2.keys()))

        # Calculate document frequency (df) for each word
        df = defaultdict(int)
        for word in unique_words:
            if word in freq1:
                df[word] += 1
            if word in freq2:
                df[word] += 1

        # Calculate inverse document frequency (idf) for each word
        idf = {}
        for word in unique_words:
            idf[word] = math.log((2+self.alpha) / (df[word]+self.alpha)) + 1

        # Calculate the numerator of the CIDEr score
        numerator = 0
        for word in unique_words:
            tf1 = freq1[word]
            tf2 = freq2[word]
            numerator += tf1 * tf2 * idf[word]**2

        # Calculate the denominator of the CIDEr score
        denominator1 = sum(tf1 * idf[word] for word, tf1 in freq1.items())
        denominator2 = sum(tf2 * idf[word] for word, tf2 in freq2.items())
        denominator = denominator1 * denominator2

        # Calculate the CIDEr score
        score = numerator / denominator**(1/2)

        return score
