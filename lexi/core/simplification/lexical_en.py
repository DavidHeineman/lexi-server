import re
import torch
import jsonpickle
import logging
import pickle
import nltk
import numpy as np
from abc import ABCMeta, abstractmethod
from lemminflect import getInflection
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 
from lexi.config import RESOURCES, MODELS_DIR, DEFAULT_THRESHOLD, \
    NUM_REPLACEMENTS, SCORER_PATH_TEMPLATE, SCORER_MODEL_PATH_TEMPLATE, \
    CWI_PATH_TEMPLATE, NRR_PATH_TEMPLATE, NRR_MODEL_PATH_TEMPLATE, \
    RANKER_PATH_TEMPLATE
from lexi.core.util import util
from lexi.core.simplification import SimplificationPipeline
from nltk.tokenize import TreebankWordTokenizer as twt
logger = logging.getLogger('lexi')

# For Ranker
from lexi.core.en_nrr.nrr import NRR

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

        # substitution counter for debugging
        count_sub = 0
        count_nosub = 0
        targ_amount = 0
        amount = []

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

                targ_amount += 1

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

                # STEP 3: SUBSTITUTION SELECTION
                selected = self.selector.select(sent, wb, we, candidates)
                if len(selected) == 0:
                    logger.debug("No ngram representations found "
                                 "for '{}' replacements.".format(sent[wb:we]))
                    count_nosub += 1
                else:
                    candidates = selected
                    count_sub += 1
                    amount.append(len(selected))
                
                # STEP 4: RANKING
                if ranker:
                    ranking = ranker.rank(candidates, sent, wb, we)
                elif self.ranker:
                    ranking = self.ranker.rank(candidates, sent, wb, we)
                else:
                    ranking = candidates
                offset2simplification[global_word_offset_start] = \
                    (sent[wb:we], ranking, sent, wb, we)             
        # Debugging Substitution Selection
        logger.info("Identified {} of {} words as complex | {}{}".format((count_sub + count_nosub), targ_amount, ((count_sub + count_nosub) / targ_amount) * 100, "%"))
        logger.info("Found n-gram representations for {} of {} substitutions | {}{}".format(count_sub, (count_sub + count_nosub), (count_sub / (count_sub + count_nosub)) * 100, "%"))
        logger.info("Average amount of n-gram representations identified: {} | {}{}".format(sum(amount) / len(amount), (sum(amount) / len(amount)) * 10, "%"))
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
        self.scorer.save()

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
    def __init__(self, userId, featurizer, model):
        self.userId = userId
        self.path = SCORER_PATH_TEMPLATE.format(userId)
        self.featurizer = featurizer
        self.model = model
        # pickle.load(open(SCORER_PATH_TEMPLATE.format(self.userId), 'rb'))
        self.model_path = SCORER_MODEL_PATH_TEMPLATE.format(self.userId)
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

    def train_model(self, x, y, epochs=10, patience=5):
        x_in = self.featurizer.featurize(x)
        y_in = np.array(y).reshape([-1, 1])
        logger.info("Fit new data to model: {}".format(str(x)[:300]))
        self.model.partial_fit(x_in, y_in.reshape(-1))

    def update(self, data):
        x, y = [], []
        for (sent, so, eo), label in data:
            x.append((sent[so:eo], sent, so, eo))
            y.append(label)
        self.train_model(x, y)

    def build_model(self):
        return

    def score(self, sent, start_offset, end_offset):
        cached = self.cache.get((sent, start_offset, end_offset))
        if cached is not None:
            return cached
        item = (sent[start_offset:end_offset], sent, start_offset, end_offset)
        x = self.featurizer.featurize([item])[0]
        score = self.model.predict([x])[0]
        self.cache[(sent, start_offset, end_offset)] = score
        return score

    def predict(self, x):
        return [self.model.predict(x)]

    def save(self):
        # save state of this object, except model (excluded in __getstate__())
        with open(self.get_path(), 'w') as f:
            json = jsonpickle.encode(self)
            f.write(json)
        # save model
        pickle.dump(self.model, open(self.model_path, 'wb'))

    @staticmethod
    def staticload(path):
        with open(path) as jsonfile:
            json = jsonfile.read()
        scorer = jsonpickle.decode(json)
        scorer.model = pickle.load(open(scorer.model_path, 'rb'))
        scorer.cache = {}
        return scorer

class MounicaScorerOLD:
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
        self.softmax = torch.nn.Softmax(dim=0)
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
        laststep = torch.sigmoid(self.out(h))
        return self.softmax(h)

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
                    # print("BEST LOSS: {}".format(float(best_loss)))
                    # print("current params:")
                    # print(list(self.parameters()))
                    self.load_state_dict(best_model)
                    # print("best params, loaded:")
                    # print(list(self.parameters()))
                    return
            loss.backward()
            optimizer.step()
            # print(list(self.parameters()))

class MounicaSelector:
    def __init__(self, ngram, google_freq_file=RESOURCES['en']['nrr']['google'], cutoff=0):
        google_freq = {}
        total = 0
        nextone = 0
        logger.debug("Loading google %d-gram frequencies..." % ngram)
        for line in open(google_freq_file, encoding='utf-8'):
            line_tokens = [t.strip() for t in line.strip().split('\t')]
            try:
                count = int(line_tokens[1])
                if count > cutoff:
                    google_freq[line_tokens[0]] = np.log10(count)
                    total += 1
                    nextone = 0
            except IndexError:
                logger.debug("Error: the following has no corresponding word: " + str(line_tokens))
                pass
            if (total % 1000 == 0 and nextone == 0):
                nextone = 1
                logger.debug("N-gram count: " + str(total))
        logger.info("Total n-grams loaded: " + str(total))
        self.ngram = ngram
        self.google_freq = google_freq
        self.ps = PorterStemmer()
        self.lem = nltk.WordNetLemmatizer()

    def select(self, sent, so, eo, candidates):
        cand = list(candidates)
        scores = self.get_scores(sent, so, eo, cand)
        out = []
        for i in range(0, len(scores)):
            if (scores[i] != 0):
                out.append(cand[i])

        # This can filter out wrong tenses & duplicates before OR after ngram comparison
        out = self.filter_out_tense(sent, so, eo, out)

        return out

    def get_scores(self, sent, so, eo, candidates):
        t_b = word_tokenize(sent[:so])
        t_a = word_tokenize(sent[eo:])
        
        if len(t_b) < self.ngram - 1:
            t_b = ['<S>'] + t_b
            
        if len(t_a) < self.ngram - 1:
            t_a = t_a + ['</S>']

        scores = []
        for word in candidates:
            combos = t_b[-self.ngram + 1:] + [word] + t_a[:self.ngram - 1]
            scores.append(0)
            for j in range(0, len(combos) - self.ngram + 1):
                phrase = ''
                for word in combos[j:j + self.ngram]:
                    phrase += word + ' '
                phrase = phrase.lower()
                if phrase[:-1] in self.google_freq:
                    scores[-1] += self.google_freq[phrase[:-1]]
        return scores

    def filter_out_tense(self, sent, so, eo, candidates):
        stems = []
        out = []
        word_tag = nltk.pos_tag([sent[so:eo]])[0][1]
        stems.append(self.ps.stem(sent[so:eo]))
        for word in candidates:
            cand_stem = self.ps.stem(word)
            if cand_stem not in stems:
                stems.append(cand_stem)
                try:
                    cand_tag = self.tag_for_lemmatizer(word)
                    if cand_tag is None:
                        out.append(getInflection(self.lem.lemmatize(word, pos=cand_tag), tag=word_tag)[0])
                    else:
                        out.append(word)
                except IndexError:
                    # Lemminflect does not support all POS tags - lemminflect.readthedocs.io/en/latest/tags/
                    out.append(word)
                    logger.debug("ERROR: Lemminflect cannot convert {} with type {}, skipping".format(word, word_tag))
        return out

    def tag_for_lemmatizer(self, word):
        tag = nltk.pos_tag([word])[0][1][:2]
        if tag in ['VB']:
            return 'v'
        elif tag in ['JJ']:
            return 'a'
        elif tag in ['RB']:
            return 'r'
        else:
            return 'n'
        
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
            return list(self.corpus[word].keys())
        else:
            return None
        
class MounicaRanker:
    def __init__(self, userId, nrr=None):
        self.userId = userId
        self.nrr = nrr
        if self.nrr is not None:
            self.nrr_path = nrr.path
       
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['nrr']
        return state

    def set_nrr(self, nrr):
        self.nrr = nrr
        self.nrr_path = nrr.get_path()

    def set_userId(self, userId):
        self.userId = userId

    def update(self, data):
        if self.nrr is not None:
            self.nrr.update(data)
    
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

    def save(self, userId):
        json = jsonpickle.encode(self)
        with open(RANKER_PATH_TEMPLATE.format(userId), 'w') as jsonfile:
            jsonfile.write(json)
        self.nrr.save()

    @staticmethod
    def staticload(path, featurizer):
        with open(path) as jsonfile:
            json = jsonfile.read()
        ranker = jsonpickle.decode(json)
        if hasattr(ranker, "nrr_path") and ranker.nrr_path is not None:
            ranker.set_nrr(MounicaNRR.staticload(ranker.nrr_path, featurizer))
        else:
            logger.warn("Ranker file does not provide link to a NRR. Set "
                        "manually with ranker.set_nrr()!")
        return ranker

class MounicaNRR:
    def __init__(self, userId, featurizer, dimensionality=600):
        self.userId = userId
        self.path = NRR_PATH_TEMPLATE.format(userId)
        self.dimensionality = dimensionality
        self.featurizer = featurizer
        self.model_path = NRR_MODEL_PATH_TEMPLATE.format(userId)
        self.model = self.build_model()
        self.update_steps = 0
        self.cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model'], state['cache'], state['featurizer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cache = {}
        self.model = self.build_model()

    def get_path(self):
        return NRR_PATH_TEMPLATE.format(self.userId)

    def set_userId(self, userId):
        self.userId = userId
        self.path = SCORER_PATH_TEMPLATE.format(userId)

    def build_model(self):
        return NRR(self.dimensionality) # self.featurizer.dimensions()

    def update(self, data):
        #for (sentence, start_offset, end_offset), label in data:
        #    self.featurizer.get_features_single
        #self.model.train(feedback_x, feedback_y, 100, 0.0005)
        #self.update_steps += 1
        return

    def train_model(self, x, y, epochs, lr):
        self.model.train(x, y, epochs, lr)
        
    def evaluate(self, sent, target, candidates):
        # Extracts features from input
        features = self.featurizer.get_features_single(sent, target, list(candidates))
        # Uses trianed input NRR to predict scores
        prediction_scores = [score[0] for score in self.model.predict(features).data.numpy()]
        # Sorts output prediction scores
        count = -1
        substitutes = candidates
        score_map = {}
        for sub in substitutes:
            score_map[sub] = 0.0
        for s1 in substitutes:
            for s2 in substitutes:
                if s1 != s2:
                    count += 1
                    score = prediction_scores[count]
                    score_map[s1] += score
        return sorted(score_map.keys(), key=score_map.__getitem__)

    def save(self):
        # save state of this object, except model (excluded in __getstate__())
        with open(self.get_path(), 'w') as f:
            json = jsonpickle.encode(self)
            f.write(json)
        # save model
        torch.save(self.model, self.model_path)

    @staticmethod
    def staticload(path, featurizer):
        with open(path) as jsonfile:
            json = jsonfile.read()
        nrr = jsonpickle.decode(json)
        nrr.cache = {}
        nrr.model = nrr.build_model()
        nrr.featurizer = featurizer
        return nrr

#### NEW PHRASAL MODELS FOR DATA COLLECTION - SHOULD EVENTUALLY BE USED IN MODEL
class MounicaGeneratorPhrasal:
    def __init__(self, ppdb_file=RESOURCES["en"]["ppdb-lexicon"], ppdb_phrasal_file=RESOURCES["en"]["ppdb-lexicon-phrasal"], language="en"):
        self.language = language
        self.corpusSingle = {}
        self.corpusPhrasal = {}
        self.tokenizer = twt()

        # Open corpus for singular words
        for line in open(ppdb_file, encoding='utf-8'):
            tokens = [t.strip() for t in line.strip().split('\t')]
            if float(tokens[2]) > 0:
                if tokens[0] not in self.corpusSingle:
                    replacements = {}
                else:
                    replacements = self.corpusSingle[tokens[0]]
                if tokens[0] not in self.corpusSingle or len(self.corpusSingle[tokens[0]]) < NUM_REPLACEMENTS:
                    replacements[tokens[1]] = float(tokens[2])
                    self.corpusSingle[tokens[0]] = replacements
        for key in self.corpusSingle.keys():
            self.corpusSingle[key] = dict(sorted(self.corpusSingle[key].items(), key=lambda item: item[1], reverse=True))

        # Open corpus for phrases
        for line in open(ppdb_phrasal_file, encoding='utf-8'):
            tokens = [t.strip() for t in line.strip().split('\t')]
            if float(tokens[2]) > 0:
                if tokens[0] not in self.corpusPhrasal:
                    replacements = {}
                else:
                    replacements = self.corpusPhrasal[tokens[0]]
                if tokens[0] not in self.corpusPhrasal or len(self.corpusPhrasal[tokens[0]]) < NUM_REPLACEMENTS:
                    replacements[tokens[1]] = float(tokens[2])
                    self.corpusPhrasal[tokens[0]] = replacements
        for key in self.corpusPhrasal.keys():
            self.corpusPhrasal[key] = dict(sorted(self.corpusPhrasal[key].items(), key=lambda item: item[1], reverse=True))

    def factorizePhrase(self, sent, so, eo):
        t_b = list((sent[wb:we], wb, we) for (wb, we) in list(self.tokenizer.span_tokenize(sent[:so])))
        t_a = list((sent[wb:we], wb, we) for (wb, we) in list((wb+eo, we+eo) for (wb, we) in self.tokenizer.span_tokenize(sent[eo:])))
        t_b = [('<S>', 0, 0)] + t_b
        t_a = t_a + [('</S>', len(sent)- 1, len(sent) - 1)]

        ngrams = []
        for i in range(0,8):
            for j in range(0,8):
                out = sent[so:eo]
                rb = so
                re = eo
                if (i != 0):
                    for word in t_b[::-1][:i]:
                        out = word[0] + " " + out
                        rb = word[1]
                if (j != 0):
                    for word in t_a[:j]:
                        out = out + " " + word[0]
                        re = word[2]
                if (len(word_tokenize(out)) < 8 and out not in ngrams):
                    ngrams.append((out, rb, re))
        return ngrams

    def getSubstitutions(self, sent, wb, we):
        output = list()
        if sent[wb:we].lower() in self.corpusSingle:
            for word in list(self.corpusSingle[sent[wb:we].lower()].keys()):
                output.append((wb, we, word, self.corpusSingle[sent[wb:we].lower()][word]))
        
        ngrams = self.factorizePhrase(sent, wb, we)
        for phrase, rb, re in ngrams:
            if phrase.lower() in self.corpusPhrasal:
                for genPhrase in list(self.corpusPhrasal[phrase.lower()].keys()):
                    output.append((rb, re, genPhrase, self.corpusPhrasal[phrase.lower()][genPhrase]))
        
        output.sort(key = lambda x: x[3], reverse=True)
        output = [(a,b,c) for (a,b,c,d) in output][:10]

        if len(output) == 0:
            return None
        return output

class MounicaSelectorPhrasal:
    def __init__(self, ngram, google_freq_file=RESOURCES['en']['nrr']['google'], cutoff=0):
        self.hi = 1
    
    def select(self, sent, so, eo, candidates):
        return candidates[:10]

class MounicaSelectorPhrasalTEMP:
    def __init__(self, ngram, google_freq_file=RESOURCES['en']['nrr']['google'], cutoff=0):
        google_freq = {}
        total = 0
        nextone = 0
        logger.debug("Loading google %d-gram frequencies..." % ngram)
        for line in open(google_freq_file, encoding='utf-8'):
            line_tokens = [t.strip() for t in line.strip().split('\t')]
            try:
                count = int(line_tokens[1])
                if count > cutoff:
                    google_freq[line_tokens[0]] = np.log10(count)
                    total += 1
                    nextone = 0
            except IndexError:
                logger.debug("Error: the following has no corresponding word: " + str(line_tokens))
                pass
            if (total % 1000 == 0 and nextone == 0):
                nextone = 1
                logger.debug("N-gram count: " + str(total))
        logger.info("Total n-grams loaded: " + str(total))
        self.ngram = ngram
        self.google_freq = google_freq
        self.ps = PorterStemmer()
        self.lem = nltk.WordNetLemmatizer()

    def select(self, sent, so, eo, candidates):
        cand = list(candidates)
        scores = self.get_scores(sent, so, eo, cand)
        out = []
        for i in range(0, len(scores)):
            if (scores[i] != 0):
                out.append(cand[i])

        # This can filter out wrong tenses & duplicates before OR after ngram comparison
        out = self.filter_out_tense(sent, so, eo, out)

        return out

    def get_scores(self, sent, so, eo, candidates):
        t_b = word_tokenize(sent[:so])
        t_a = word_tokenize(sent[eo:])
        
        if len(t_b) < self.ngram - 1:
            t_b = ['<S>'] + t_b
            
        if len(t_a) < self.ngram - 1:
            t_a = t_a + ['</S>']

        scores = []
        for word in candidates:
            combos = t_b[-self.ngram + 1:] + [word] + t_a[:self.ngram - 1]
            scores.append(0)
            for j in range(0, len(combos) - self.ngram + 1):
                phrase = ''
                for word in combos[j:j + self.ngram]:
                    phrase += word + ' '
                phrase = phrase.lower()
                if phrase[:-1] in self.google_freq:
                    scores[-1] += self.google_freq[phrase[:-1]]
        return scores

    def filter_out_tense(self, sent, so, eo, candidates):
        stems = []
        out = []
        word_tag = nltk.pos_tag([sent[so:eo]])[0][1]
        stems.append(self.ps.stem(sent[so:eo]))
        for word in candidates:
            cand_stem = self.ps.stem(word)
            if cand_stem not in stems:
                stems.append(cand_stem)
                try:
                    cand_tag = self.tag_for_lemmatizer(word)
                    if cand_tag is None:
                        out.append(getInflection(self.lem.lemmatize(word, pos=cand_tag), tag=word_tag)[0])
                    else:
                        out.append(word)
                except IndexError:
                    # Lemminflect does not support all POS tags - lemminflect.readthedocs.io/en/latest/tags/
                    out.append(word)
                    logger.debug("ERROR: Lemminflect cannot convert {} with type {}, skipping".format(word, word_tag))
        return out

    def tag_for_lemmatizer(self, word):
        tag = nltk.pos_tag([word])[0][1][:2]
        if tag in ['VB']:
            return 'v'
        elif tag in ['JJ']:
            return 'a'
        elif tag in ['RB']:
            return 'r'
        else:
            return 'n'