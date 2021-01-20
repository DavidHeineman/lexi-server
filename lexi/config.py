import os

SOURCE_BASE = os.path.dirname(os.path.realpath(__file__))
LEXI_BASE = os.path.join(SOURCE_BASE, "..")
LOG_DIR = os.path.join(LEXI_BASE, "logs")
MODELS_DIR = os.path.join(LEXI_BASE, "models")
RANKER_DIR = os.path.join(MODELS_DIR, "rankers")
CWI_DIR = os.path.join(MODELS_DIR, "cwi")
SCORERS_DIR = os.path.join(MODELS_DIR, "scorers")
FEATURIZERS_DIR = os.path.join(MODELS_DIR, "featurizers")
RESOURCES_DIR = os.path.join(LEXI_BASE, "res")
STANFORDNLP = os.path.join(RESOURCES_DIR, "stanfordnlp_resources")

RANKER_PATH_TEMPLATE = os.path.join(RANKER_DIR, "{}.json")
CWI_PATH_TEMPLATE = os.path.join(CWI_DIR, "{}.json")
SCORER_PATH_TEMPLATE = os.path.join(SCORERS_DIR, "{}.json")
SCORER_MODEL_PATH_TEMPLATE = os.path.join(SCORERS_DIR, "{}.pt")
FEATURIZER_PATH_TEMPLATE = os.path.join(FEATURIZERS_DIR, "{}.json")

LEXICAL_MODEL_PATH_TEMPLATE = os.path.join(MODELS_DIR, "{}-lexical.pickle")
MODEL_PATH_TEMPLATE = os.path.join(MODELS_DIR, "{}.pickle")

RESOURCES = {
    "en": {
        #"embeddings":
        #    [RESOURCES_DIR + "/en/en_embedding.bin"],
        #"lm":
        #    RESOURCES_DIR + "/en/en_lm.bin",
        #"ubr":
        #    RESOURCES_DIR + "/en/en_ubr.txt",
        #"ranking_training_dataset":
        #    RESOURCES_DIR + "/en/en_ls.txt",
        #"synonyms":
        #    [RESOURCES_DIR + "/en/en_synonyms.csv"],
        "mounica-lexicon":
            RESOURCES_DIR + "/en/complexity_lexicon.tsv",
        "ppdb-lexicon":
            RESOURCES_DIR + "/en/simpleppdbpp_lexicon.txt",
        "nrr": {
            "train": RESOURCES_DIR + "/nrr/test_data_victor.txt",
            "test": RESOURCES_DIR + "/nrr/train_data_victor.txt",
            "lm": RESOURCES_DIR + "/nrr/subimdb_5_srilm_default.bin",
            "lexicon": RESOURCES_DIR + "/nrr/lexicon.tsv",
            "word2vec": RESOURCES_DIR + "/nrr/en_googlenews_embedding.bin",
            "google": RESOURCES_DIR + "/nrr/semeval_train_test_google_freq.txt",
            "wiki": RESOURCES_DIR + "/nrr/simpleppdb_human_wiki_ratio.txt",
            "ppdb": RESOURCES_DIR + "/nrr/semeval_train_test_ppdb_scores.txt",
            
            #"model": MODELS_DIR + "/mounica/nrr_default.bin",
        }
    },
        
}