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