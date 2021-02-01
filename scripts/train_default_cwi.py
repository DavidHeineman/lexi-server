from lexi.core.simplification.lexical_en import *
from lexi.core.featurize.featurizers import LexicalFeaturizer
from lexi.core.featurize.functions_en import ComplexityLexicon
from lexi.core.featurize.functions import *
from lexi.config import FEATURIZER_PATH_TEMPLATE, RESOURCES, FEATURIZERS_DIR, \
    CWI_DIR, SCORERS_DIR, DEFAULT_THRESHOLD, CWI_PATH_TEMPLATE
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from scipy.stats.stats import spearmanr
import sys
import os
import pickle

# Ignore warnings about softmax (fix this)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

CWI_MODEL = 'mlp' # mlp or pytorch

def main():
    resources = RESOURCES['en']

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
                s = line[1]           # Sentence
                so = int(line[2])     # Start Offset
                eo = int(line[3])     # End Offset
                w = s[so:eo]          # Word
                items.append((w, s, so, eo))
                y.append(int(line[9]))
    except UnicodeDecodeError:
        pass

    x = lf.featurize(items, fit=True, scale_features=True)
    y = np.array(y).reshape([-1, 1])
    
    # Comparison between custom torch model and sci-kit learn model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print("Training MLP Classifier w/ Train/Test split...")
    mlp = MLPClassifier(max_iter=1000, warm_start=True, hidden_layer_sizes=[10])
    mlp.fit(x_train, y_train.reshape(-1))
    p = mlp.predict(x_test)
    print("F1: %.4f | Accuracy: %.4f" % (f1_score(y_test, p), accuracy_score(y_test, p)))

    print("Training custom pytorch w/ Train/Test split...")
    ls = MounicaScorerOLD("default", lf, [])
    ls.train_model(x_train, y_train, epochs=1000, patience=10)
    p = np.array(ls.predict(x_test))
    print("F1: %.4f | Accuracy: %.4f" % (f1_score(y_test, p), accuracy_score(y_test, p)))
    
    # We train the models again with all the data because we're not measuring their performance this time
    if CWI_MODEL == 'mlp':
        print("Training MLP Classifier to be used as the scorer...")
        mlp = MLPClassifier(max_iter=1000, warm_start=True, hidden_layer_sizes=[10])
        mlp.fit(x, y.reshape(-1))
        p = mlp.predict(x)
        print("Done!")

        mlp_ls = MounicaScorer("default", lf, mlp)
        mlp_ls.save()
        print('Saved MLP scorer model to %s' % (SCORERS_DIR+'\default.json'))
    elif CWI_MODEL == 'pytorch':
        print("Training using custom pytorch model...")
        ls = MounicaScorer("defaultOLD", lf, [])
        ls.train_model(x, y, epochs=1000, patience=10)
        p = np.array(ls.predict(x))
        print("Done!")
        
        if not os.path.exists(SCORERS_DIR):
            os.makedirs(SCORERS_DIR)
        ls.save()
        print('Saved pytorch scorer model to %s' % (SCORERS_DIR+'\default.json'))

    # Save featurizer
    if not os.path.exists(FEATURIZERS_DIR):
        os.makedirs(FEATURIZERS_DIR)
    lf.save(FEATURIZER_PATH_TEMPLATE.format("default"))
    print('Saved featurizer to %s' % (FEATURIZERS_DIR+'\default.json'))

    # Configure and save CWI
    cwi = MounicaCWI("default")
    cwi.set_scorer(MounicaScorer.staticload(SCORER_PATH_TEMPLATE.format("default")))
    cwi.set_cwi_threshold(DEFAULT_THRESHOLD)

    if not os.path.exists(CWI_DIR):
        os.makedirs(CWI_DIR)
    print('Saved CWI to %s' % (CWI_DIR+'\default.json'))
    cwi.save("default")

if __name__ == '__main__':
    main()