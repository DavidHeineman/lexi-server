from lexi.core.simplification.lexical_en import *
from lexi.core.featurize.featurizers import LexicalFeaturizer
from lexi.core.featurize.functions_en import ComplexityLexicon
from lexi.core.featurize.functions import *
from lexi.config import FEATURIZER_PATH_TEMPLATE, RESOURCES, FEATURIZERS_DIR, \
    CWI_DIR, SCORERS_DIR, DEFAULT_THRESHOLD, CWI_PATH_TEMPLATE
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from scipy.stats.stats import spearmanr
import sys
import os

def main():
    lf = LexicalFeaturizer()
    lf.add_feature_function(WordLength())
    lf.add_feature_function(SentenceLength())
    lf.add_feature_function(IsAlpha())
    lf.add_feature_function(IsLower())
    lf.add_feature_function(IsNumerical())
    lf.add_feature_function(ComplexityLexicon())

    items, y = [], []

    try:
        for line in open(RESOURCES['en']['cwi']['train'], encoding='utf-8'):
            line = line.strip().split("\t")
            if line:
                # line[0] contains the sentence
                # print(line[4] + "\t\t" + line[9] + "\t" + line[10])
                s = line[1]           # Sentence
                so = int(line[2])     # Start Offset
                eo = int(line[3])     # End Offset
                w = s[so:eo]          # Word
                items.append((w, s, so, eo))
                y.append(int(line[9]))
    except UnicodeDecodeError:
        pass

    x = lf.featurize(items, fit=True, scale_features=True)

    if not os.path.exists(FEATURIZERS_DIR):
        print('Saved featurizer to %s' % (FEATURIZERS_DIR+'/default.json'))
        os.makedirs(FEATURIZERS_DIR)

    lf.save(FEATURIZER_PATH_TEMPLATE.format("default"))
    y = np.array(y).reshape([-1, 1])

    ls = MounicaScorer("default", lf, [])
    ls.train_model(x, y, epochs=1000, patience=10)
    p = np.array(ls.predict(x))

    if not os.path.exists(SCORERS_DIR):
        print('Saved model to %s' % (SCORERS_DIR+'/default.json'))
        os.makedirs(SCORERS_DIR)

    ls.save()

    mlp = MLPRegressor(max_iter=1000, warm_start=True, hidden_layer_sizes=[10], verbose=False)
    mlp.fit(x, y.reshape(-1))
    p = mlp.predict(x)

    language = 'en'
    c = MounicaSimplificationPipeline(userId='default', language=language)
    try:
        resources = RESOURCES[language]
    except KeyError:
        print("Couldn't find resources for language {}".format(language))
    c.setCwi(MounicaCWI("default"))
    c.cwi.set_scorer(MounicaScorer.staticload(SCORER_PATH_TEMPLATE.format("default")))
    c.cwi.set_cwi_threshold(DEFAULT_THRESHOLD)

    if not os.path.exists(CWI_DIR):
        print('Saved model to %s' % (CWI_DIR+'/default.json'))
        os.makedirs(CWI_DIR)
    c.cwi.save("default")

if __name__ == '__main__':
    main()