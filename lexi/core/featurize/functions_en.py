from abc import ABCMeta, abstractmethod
import numpy as np
from lexi.core.simplification.lexical_en import WordComplexityLexicon
from lexi.core.featurize.functions import FeatureFunction
from lexi.config import RESOURCES, DEFAULT_THRESHOLD

# # #  Custom English Feature Functions
  
class ComplexityLexicon(FeatureFunction):

    def __init__(self, name="word_length"):
        super().__init__(name)
        self.lexicon = WordComplexityLexicon(RESOURCES["en"]["mounica-lexicon"])
        self.cwi_threshold = DEFAULT_THRESHOLD

    def process(self, word, sentence, startOffset, endOffset):
        return float(self.lexicon.get_feature([sentence[startOffset:endOffset]])[0] / 6) > self.cwi_threshold