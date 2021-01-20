from nltk import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

from lexi.config import RESOURCES, MODELS_DIR

from lexi.core.util import util
from lexi.core.simplification import SimplificationPipeline
import logging
logger = logging.getLogger('lexi')

from lexi.core.en_nrr.evaluator import SingleNRR

import re

class MounicaSimplificationPipeline(SimplificationPipeline):

    def __init__(self, userId, language="da"):
        self.language = language
        self.userId = userId
        self.cwi = None
        self.generator = None
        self.selector = None
        self.ranker = None

    def generateCandidates(self, sent, startOffset, endOffset,
                           min_similarity=0.6):
        if self.generator is not None:
            return self.generator.getSubstitutions(
                sent[startOffset:endOffset], min_similarity=min_similarity)
        return []

    def selectCandidates(self, sent, startOffset, endOffset, candidates):
        if self.selector is not None:
            return self.selector.select(sent, startOffset, endOffset,
                                        candidates)
        return candidates  # fallback if selector not set

    def setCwi(self, cwi):
        self.cwi = cwi

    def setRanker(self, ranker):
        self.ranker = ranker

    def setGenerator(self, generator):
        self.generator = generator

    def setSelector(self, selector):
        self.selector = selector

    def simplify_text(self, text, startOffset=0, endOffset=None, cwi=None,
                      ranker=None, min_similarity=0.6, blacklist=None):
        """
        Full lexical simplification pipeline.
        :param text:
        :param startOffset:
        :param endOffset:
        :param cwi:
        :param ranker:
        :param min_similarity:
        :param blacklist:
        :return:
        """
        startOffset = max(0, startOffset)
        endOffset = min(len(text), endOffset)

        offset2simplification = {}
        sent_offsets = list(util.span_tokenize_sents(text))
        logger.debug("Sentences: {}".format(sent_offsets))
        # word_offsets = util.span_tokenize_words(pure_text)
        for sb, se in sent_offsets:
            # ignore all sentences that end before the selection or start
            # after the selection
            if se < startOffset or sb > endOffset:
                continue

            sent = text[sb:se]
            token_offsets = util.span_tokenize_words(sent)

            for wb, we in token_offsets:
                global_word_offset_start = sb + wb
                global_word_offset_end = sb + we
                if global_word_offset_start < startOffset or \
                        global_word_offset_end > endOffset:
                    continue

                # STEP 1: TARGET IDENTIFICATION
                complex_word = True  # default case, e.g. for when no CWI module
                # provided for single-word requests
                if cwi:
                    complex_word = cwi.is_complex(sent, wb, we)
                elif self.cwi:
                    complex_word = self.cwi.is_complex(sent, wb, we)
                if not complex_word:
                    continue

                logger.debug("Identified targets: {}".format(sent[wb:we]))

                # STEP 2: CANDIDATE GENERATION
                candidates = self.generator.getSubstitutions(sent[wb:we])
                if not candidates:
                    logger.debug("No candidate replacements found "
                                 "for '{}'.".format(sent[wb:we]))
                    continue
                logger.debug("Candidate replacements: {}.".format(candidates))
                
                # STEP 4: RANKING
                if ranker:
                    ranking = ranker.rank(candidates, sent, wb, we)
                elif self.ranker:
                    ranking = self.ranker.rank(candidates, sent, wb, we)
                else:
                    ranking = candidates
                offset2simplification[global_word_offset_start] = \
                    (sent[wb:we], ranking, sent, wb, we)             
        return offset2simplification

class MounicaCWI():
    def __init__(self):
        self.lexicon = WordComplexityLexicon(RESOURCES["en"]["mounica-lexicon"])
        self.cwi_threshold = 0.5
        
    def score(self, sent, start_offset, end_offset):
        return float(self.lexicon.get_feature([sent[start_offset:end_offset]])[0] / 6)
    
    def identify_targets(self, sent, token_offsets):
        return [(wb, we) for wb, we in token_offsets if
                self.is_complex(sent, wb, we)]
    
    def is_complex(self, sent, startOffset, endOffset):
        cwi_score = self.score(sent, startOffset, endOffset)
        return cwi_score > self.cwi_threshold
    
    def set_cwi_threshold(self, threshold):
        self.cwi_threshold = threshold
        
        
class MounicaGenerator:
    def __init__(self, ppdb_file=RESOURCES["en"]["ppdb-lexicon"], language="en"):
        self.language = language
        self.corpus = {}
        for line in open(ppdb_file, encoding='utf-8'):
            tokens = [t.strip() for t in line.strip().split('\t')]
            if float(tokens[2]) > 0:
                if tokens[0] not in self.corpus:
                    replacements = {}
                else:
                    replacements = self.corpus[tokens[0]]
                if tokens[0] not in self.corpus or len(self.corpus[tokens[0]]) < 10:
                    replacements[tokens[1]] = float(tokens[2])
                    self.corpus[tokens[0]] = replacements

        for key in self.corpus.keys():
            self.corpus[key] = dict(sorted(self.corpus[key].items(), key=lambda item: item[1], reverse=True))
            
    def getSubstitutions(self, word):
        if word in self.corpus:
            return set(self.corpus[word].keys())
        else:
            return None
        
class MounicaRanker:
    def __init__(self, resources=RESOURCES):
        self.nrr = SingleNRR(resources['en']['nrr'], MODELS_DIR+'/rankers/default.bin')
        
    def rank(self, candidates, sentence=None, wb=0, we=0):
        try:
            output = self.nrr.evaluate(sentence, sentence[wb:we], candidates)
        except ValueError:
            # Really janky error handling - the ngram model for nrr doesn't work with punctuation...
            logger.debug("Error in replacement: {} for sentence {}.".format(sentence, sentence[wb:we]))
            try:
                output = self.nrr.evaluate(re.sub('[^a-zA-Z ]', " ", sentence), sentence[wb:we], candidates)
            except IndexError:
                output = candidates
        except IndexError:
            output = candidates
        return output

class WordComplexityLexicon:
    def __init__(self, lexicon):
        word_ratings = {}
        for line in open(lexicon):
            tokens = [t.strip() for t in line.strip().split('\t')]
            word_ratings[tokens[0].lower()] = float(tokens[1])
        self.word_ratings = word_ratings
        self.lemmatizer = WordNetLemmatizer()
        self.lancaster_stemmer = LancasterStemmer(strip_prefix_flag=True)
        self.snowball_stemmer = SnowballStemmer("english")

    def get_feature(self, words):
        phrase = max(words, key=len)

        if phrase in self.word_ratings:
            return [self.word_ratings[phrase], 1.0]
        else:
            ratings = []
            lemman = self.lemmatizer.lemmatize(phrase, pos='n')
            lemmav = self.lemmatizer.lemmatize(phrase, pos='v')
            lemmaa = self.lemmatizer.lemmatize(phrase, pos='a')
            lemmar = self.lemmatizer.lemmatize(phrase, pos='r')
            stem_lan = self.lancaster_stemmer.stem(phrase)
            stem_snow = self.snowball_stemmer.stem(phrase)

            if lemman in self.word_ratings:
                ratings.append(self.word_ratings[lemman])
            elif lemmav in self.word_ratings:
                ratings.append(self.word_ratings[lemmav])
            elif lemmaa in self.word_ratings:
                ratings.append(self.word_ratings[lemmaa])
            elif lemmar in self.word_ratings:
                ratings.append(self.word_ratings[lemmar])
            elif stem_snow in self.word_ratings:
                ratings.append(self.word_ratings[stem_snow])
            elif stem_lan in self.word_ratings and abs(len(stem_lan) - len(phrase)) <= 2:
                ratings.append(self.word_ratings[stem_lan])

            if len(ratings) > 0:
                return [max(ratings)*1.0, 1.0]

        return [0.0, 0.0]