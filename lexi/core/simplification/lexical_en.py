import re
import torch
import jsonpickle
import logging
from abc import ABCMeta, abstractmethod
from nltk import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from lexi.config import RESOURCES, MODELS_DIR, DEFAULT_THRESHOLD, \
    NUM_REPLACEMENTS, SCORER_PATH_TEMPLATE, SCORER_MODEL_PATH_TEMPLATE, \
    CWI_PATH_TEMPLATE
from lexi.core.util import util
from lexi.core.simplification import SimplificationPipeline
from lexi.core.en_nrr.evaluator import SingleNRR
logger = logging.getLogger('lexi')


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

class MounicaPersonalizedPipelineStep(metaclass=ABCMeta):

    def __init__(self, userId=None, scorer=None):
        self.userId = userId
        self.scorer = scorer
        self.scorer_path = None

    def set_scorer(self, scorer):
        self.scorer = scorer
        self.scorer_path = scorer.get_path()

    def set_userId(self, userId):
        self.userId = userId

    @abstractmethod
    def update(self, data):
        raise NotImplementedError

    def __getstate__(self):
        """
        Needed to save pipeline steps using jsonpickle, since this module cannot
        handle torch models -- we use torch's model saving functionality
        instead. This is the method used by jsonpickle to get the state of the
        object when serializing.
        :return:
        """
        state = self.__dict__.copy()
        del state['scorer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class MounicaCWI(MounicaPersonalizedPipelineStep):
    def __init__(self, userId, scorer=None):
        super().__init__(userId, scorer)
        self.cwi_threshold = DEFAULT_THRESHOLD

    def identify_targets(self, sent, token_offsets):
        return [(wb, we) for wb, we in token_offsets if
                self.is_complex(sent, wb, we)]

    def is_complex(self, sent, startOffset, endOffset):
        if endOffset - startOffset > 2:
            cwi_score = self.scorer.score(sent, startOffset, endOffset)
            return cwi_score > self.cwi_threshold
        else:
            return False

    def set_cwi_threshold(self, threshold):
        self.cwi_threshold = threshold

    def update(self, data):
        if self.scorer:
            self.scorer.update(data)

    def save(self, userId):
        json = jsonpickle.encode(self)
        with open(CWI_PATH_TEMPLATE.format(userId), 'w') as jsonfile:
            jsonfile.write(json)

    @staticmethod
    def staticload(path):
        with open(path) as jsonfile:
            json = jsonfile.read()
        cwi = jsonpickle.decode(json)
        if hasattr(cwi, "scorer_path") and cwi.scorer_path is not None:
            cwi.set_scorer(MounicaScorer.staticload(cwi.scorer_path))
        else:
            logger.warn("CWI file does not provide link to a scorer. Set "
                        "manually with ranker.set_scorer()!")
        return cwi
        
class MounicaScorer:
    def __init__(self, userId, featurizer, hidden_dims):
        self.userId = userId
        self.path = SCORER_PATH_TEMPLATE.format(userId)
        self.featurizer = featurizer
        self.hidden_dims = hidden_dims
        self.model = self.build_model()
        self.model_path = SCORER_MODEL_PATH_TEMPLATE.format(self.userId)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=3e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=True)
        self.update_steps = 0
        self.cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model'], state['cache']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cache = {}
        self.model = self.build_model()

    def get_path(self):
        return SCORER_PATH_TEMPLATE.format(self.userId)

    def set_userId(self, userId):
        self.userId = userId
        self.path = SCORER_PATH_TEMPLATE.format(userId)

    def build_model(self):
        return LexiScorerNet(self.featurizer.dimensions(), self.hidden_dims)

    def train_model(self, x, y, epochs=10, patience=5):
        self.model.fit(torch.Tensor(x), torch.Tensor(y), self.optimizer, epochs,
                       patience)

    def update(self, data):
        # TODO do this in one batch (or several batches of more than 1 item...)
        for (sentence, start_offset, end_offset), label in data:
            x = self.featurizer.featurize(sentence, start_offset, end_offset)
            self.model.fit(x, label, self.optimizer)
            self.update_steps += 1

    def score(self, sent, start_offset, end_offset):
        cached = self.cache.get((sent, start_offset, end_offset))
        if cached is not None:
            return cached
        self.model.eval()
        item = (sent[start_offset:end_offset], sent, start_offset, end_offset)
        x = self.featurizer.featurize([item])[0]
        score = float(self.model.forward(x))
        self.cache[(sent, start_offset, end_offset)] = score
        return score

    def predict(self, x):
        return [float(self.model.forward(xi)) for xi in x]

    def save(self):
        # save state of this object, except model (excluded in __getstate__())
        with open(self.get_path(), 'w') as f:
            json = jsonpickle.encode(self)
            f.write(json)
        # save model
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, self.model_path)

    @staticmethod
    def staticload(path):
        with open(path) as jsonfile:
            json = jsonfile.read()
        scorer = jsonpickle.decode(json)
        scorer.cache = {}
        scorer.model = scorer.build_model()
        checkpoint = torch.load(scorer.model_path)
        scorer.model.load_state_dict(checkpoint['model_state_dict'])
        return scorer

class LexiScorerNet(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(LexiScorerNet, self).__init__()
        if hidden_sizes:
            self.input = torch.nn.Linear(input_size, hidden_sizes[0])
            self.hidden_layers = [torch.nn.Linear(hidden_sizes[i],
                                                  hidden_sizes[i + 1])
                                  for i in range(len(hidden_sizes) - 1)]
            self.out = torch.nn.Linear(hidden_sizes[-1], 1)
        else:
            self.input = torch.nn.Linear(input_size, 1)
            self.out = lambda x: x
            self.hidden_layers = []
        # self.apply(self.init_weights)
        # self.apply(torch.nn.init.xavier_normal_)
        # self.lossfunc = torch.nn.functional.binary_cross_entropy_with_logits

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            m.weight.data.fill_(0.5)
            m.bias.data.fill_(0.5)

    def forward(self, x):
        x = torch.Tensor(x)
        h = torch.sigmoid(self.input(x))
        for layer in self.hidden_layers:
            h = torch.torch.nn.functional.relu(layer(h))
        return torch.sigmoid(self.out(h))

    def fit(self, x, y, optimizer, epochs=100, patience=10):
        optimizer.zero_grad()
        best_loss = 1000
        best_model = None
        no_improvement_for = 0
        for _ in range(epochs):
            self.train()
            pred = self.forward(x)
            # loss = torch.sqrt(torch.mean((y - pred) ** 2))
            loss = torch.mean(torch.abs(y-pred))
            # loss = self.lossfunc(pred, y)
            # loss = torch.mean((y - pred))

            if loss < best_loss:
                best_loss = loss
                best_model = self.state_dict()
                no_improvement_for = 0
            else:
                no_improvement_for += 1
                if no_improvement_for == patience:
                    print("BEST LOSS: {}".format(float(best_loss)))
                    print("current params:")
                    print(list(self.parameters()))
                    self.load_state_dict(best_model)
                    print("best params, loaded:")
                    print(list(self.parameters()))
                    return
            print(loss)
            loss.backward()
            optimizer.step()
            # print(list(self.parameters()))

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
                if tokens[0] not in self.corpus or len(self.corpus[tokens[0]]) < NUM_REPLACEMENTS:
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
            except ValueError:
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
            try:
                stem_snow = self.snowball_stemmer.stem(phrase)
            except TypeError:
                stem_snow = ""

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