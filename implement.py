# import module
from scores import *

"""SPICE"""
spice = SpiceScore()

ref_sentence = "আমি বাংলায় গান গাই"
can_sentence = "আমি বাংলায় গান শুনি"
spice_score = spice.compute_spice(ref_sentence, can_sentence)
print("SPICE score:", spice_score)


"""METEOR"""
meteor = MeteorScore(alpha=0.5, beta=0.5, gamma=0.5)

hypothesis = "আমি বাংলায় গান গাই"
reference = "আমি বাংলায় গান শুনি"

score = meteor.meteor_score(hypothesis, reference)
print("METEOR score:", score)


"""CIDEr"""
scorer = CiderScorer(alpha=10.0)

sentence1 = "আমি বাংলায় গান গাই"
sentence2 = "আমি বাংলায় গান শুনি"

score = scorer.calculate_cider_score(sentence1, sentence2)
print("CIDEr score:", score)
