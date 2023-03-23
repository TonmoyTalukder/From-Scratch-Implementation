# -*- coding: utf-8 -*-

"""# Spice Score"""


import string

class SpiceScore:
    def __init__(self):
        pass
    
    def preprocess(self, sentence):
        # """
        # Helper function to preprocess a sentence.
        # """
        # sentence = sentence.lower()                   # convert to lowercase
        # sentence = sentence.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        # sentence = sentence.split()                    # split into words
        # return ' '.join(sentence)                      # join words back into a string
        return sentence
    
    def compute_word_matches(self, ref_sentence, can_sentence):
        """
        Helper function to compute the number of word matches between a reference sentence and a candidate sentence.
        """
        ref_sentence = set(self.preprocess(ref_sentence))
        can_sentence = set(self.preprocess(can_sentence))
        matches = len(ref_sentence.intersection(can_sentence))
        return matches
    
    def compute_precision_recall(self, ref_counts, can_counts, matches):
        """
        Helper function to compute precision and recall scores given the counts of n-grams in the reference and candidate sentences
        and the number of matching n-grams.
        """
        precision = matches / can_counts if can_counts > 0 else 0
        recall = matches / ref_counts if ref_counts > 0 else 0
        return precision, recall
    
    def compute_spice(self, ref_sentence, can_sentence):
        """
        Computes the SPICE score between a reference sentence and a candidate sentence.
        """
        ref_sentence = self.preprocess(ref_sentence)
        can_sentence = self.preprocess(can_sentence)

        # Compute the word matches between the reference and candidate sentences.
        matches = self.compute_word_matches(ref_sentence, can_sentence)

        # Compute the precision and recall scores for 1-gram to 4-gram matches.
        precisions = []
        recalls = []
        for n in range(1, 5):
            ref_counts = len(ref_sentence) - n + 1
            can_counts = len(can_sentence) - n + 1
            ngram_matches = 0
            for i in range(len(can_sentence) - n + 1):
                ngram = tuple(can_sentence[i:i+n])
                ngram_matches += int(ngram in set([tuple(ref_sentence[j:j+n]) for j in range(len(ref_sentence) - n + 1)]))
            ngram_precision, ngram_recall = self.compute_precision_recall(ref_counts, can_counts, ngram_matches)
            precisions.append(ngram_precision)
            recalls.append(ngram_recall)

        # Compute the harmonic mean of the precision and recall scores for all n-grams.
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        spice_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if avg_precision + avg_recall > 0 else 0
        return spice_score


"""# Meteor Score"""

from collections import Counter

class MeteorScore:
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def preprocess_sentence(self, sentence):
        words = sentence.split()
        return words

    def ngram_count(self, sentence, n):
        words = self.preprocess_sentence(sentence)
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        return Counter(ngrams)

    def compute_precision(self, hypothesis, reference, n):
        hyp_counts = self.ngram_count(hypothesis, n)
        ref_counts = self.ngram_count(reference, n)
        overlap = sum((hyp_counts & ref_counts).values())
        precision = overlap / sum(hyp_counts.values()) if sum(hyp_counts.values()) > 0 else 0
        return precision

    def compute_recall(self, hypothesis, reference, n):
        hyp_counts = self.ngram_count(hypothesis, n)
        ref_counts = self.ngram_count(reference, n)
        overlap = sum((hyp_counts & ref_counts).values())
        recall = overlap / sum(ref_counts.values()) if sum(ref_counts.values()) > 0 else 0
        return recall

    def meteor_score(self, hypothesis, reference):
        precision = self.alpha * self.compute_precision(hypothesis, reference, 1) + (1-self.alpha) * self.compute_precision(hypothesis, reference, 2)
        recall = self.beta * self.compute_recall(hypothesis, reference, 1) + (1-self.beta) * self.compute_recall(hypothesis, reference, 2)
        fmean = (1-self.gamma) * precision + self.gamma * recall if (precision != 0 and recall != 0) else 0
        return fmean


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