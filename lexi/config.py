import os

SOURCE_BASE = os.path.dirname(os.path.realpath(__file__))
LEXI_BASE = os.path.join(SOURCE_BASE, "..")
LOG_DIR = os.path.join(LEXI_BASE, "logs")
MODELS_DIR = os.path.join(LEXI_BASE, "models")
RANKER_DIR = os.path.join(MODELS_DIR, "rankers")
CWI_DIR = os.path.join(MODELS_DIR, "cwi")
SCORERS_DIR = os.path.join(MODELS_DIR, "scorers")
NRR_DIR = os.path.join(MODELS_DIR, "nrr")
FEATURIZERS_DIR = os.path.join(MODELS_DIR, "featurizers")
RESOURCES_DIR = os.path.join(LEXI_BASE, "res")
STANFORDNLP = os.path.join(RESOURCES_DIR, "stanfordnlp_resources")

RANKER_PATH_TEMPLATE = os.path.join(RANKER_DIR, "{}.json")
CWI_PATH_TEMPLATE = os.path.join(CWI_DIR, "{}.json")
SCORER_PATH_TEMPLATE = os.path.join(SCORERS_DIR, "{}.json")
SCORER_MODEL_PATH_TEMPLATE = os.path.join(SCORERS_DIR, "{}.pt")
NRR_PATH_TEMPLATE = os.path.join(NRR_DIR, "{}.json")
NRR_MODEL_PATH_TEMPLATE = os.path.join(NRR_DIR, "{}.bin")
FEATURIZER_PATH_TEMPLATE = os.path.join(FEATURIZERS_DIR, "{}.json")

LEXICAL_MODEL_PATH_TEMPLATE = os.path.join(MODELS_DIR, "{}-lexical.pickle")
MODEL_PATH_TEMPLATE = os.path.join(MODELS_DIR, "{}.pickle")

# Parameters for English Models
DEFAULT_THRESHOLD = 0.5     # Default threshold for determining a word "complex" for new users
# Note: This won't be needed anymore because the new CWIs are classifiers
NUM_REPLACEMENTS = 10       # Maximum number of candidate replacements generated
NGRAM = 2                   # N-Gram number of google n-gram frequencies

# See README.MD in project directory for links to download the data
RESOURCES = {
    "en": {
        "mounica-lexicon":
            RESOURCES_DIR + "/en/complexity_lexicon.tsv",
        "ppdb-lexicon":
            RESOURCES_DIR + "/en/simpleppdbpp_lexicon.txt",
        "ppdb-lexicon-phrasal": 
            RESOURCES_DIR + "/en/simpleppdbpp-phrasal.txt",
        "nrr": {
            "train": RESOURCES_DIR + "/en/nrr/test_data_victor.txt",
            "test": RESOURCES_DIR + "/en/nrr/train_data_victor.txt",
            "lm": RESOURCES_DIR + "/en/nrr/subimdb_5_srilm_default.bin",
            "lexicon": RESOURCES_DIR + "/en/complexity_lexicon.tsv",
            "word2vec": RESOURCES_DIR + "/en/nrr/en_googlenews_embedding.bin",
            "google": RESOURCES_DIR + "/en/nrr/google_freq_all.bin",
            "wiki": RESOURCES_DIR + "/en/nrr/simpleppdb_human_wiki_ratio.txt",
            "ppdb": RESOURCES_DIR + "/en/nrr/semeval_train_test_ppdb_scores.txt"
        },
        "cwi": {
            "train": RESOURCES_DIR + "/en/cwi/train_data.tsv",
        }
    },
        
}