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
        precision = overlap / sum(hyp_counts.values()
                                  ) if sum(hyp_counts.values()) > 0 else 0
        return precision

    def compute_recall(self, hypothesis, reference, n):
        hyp_counts = self.ngram_count(hypothesis, n)
        ref_counts = self.ngram_count(reference, n)
        overlap = sum((hyp_counts & ref_counts).values())
        recall = overlap / sum(ref_counts.values()
                               ) if sum(ref_counts.values()) > 0 else 0
        return recall

    def meteor_score(self, hypothesis, reference):
        precision = self.alpha * self.compute_precision(hypothesis, reference, 1) + (
            1-self.alpha) * self.compute_precision(hypothesis, reference, 2)
        recall = self.beta * self.compute_recall(hypothesis, reference, 1) + (
            1-self.beta) * self.compute_recall(hypothesis, reference, 2)
        fmean = (1-self.gamma) * precision + self.gamma * \
            recall if (precision != 0 and recall != 0) else 0
        return fmean
