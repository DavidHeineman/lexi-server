# LexiPhrasal

Implements a backend for an English version of [Lexi](https://www.aclweb.org/anthology/C18-1021.pdf) capable of phrase substitutions. It uses a different CWI and ranking model as described [in this paper](https://www.aclweb.org/anthology/D18-1410.pdf).

## Setup

- Download data to `res/en` (see below)
- Install and setup Postgres database: `lexi_schema.pgsql`
- Install requirements: `pip install -r requirements.txt`
- Train the default CWI: `python scripts/train_default_cwi.py`
- Train the default Ranker: `python scripts/train_default_ranker.py`
- Run the server with `python lexi/server/run_lexi_server.py -S`
- Install [Lexi Frontend](https://github.com/jbingel/lexi-frontend) and it will work in English.

**Note:** If you're not running Lexi on `localhost`, use `lexi.cfg` to modify the server configuration.

## Data

Data is located in the `res/en` folder. Modify `lexi/config.py` to point the project to your data.

### For CWI

- [Complexity Lexicon](https://raw.githubusercontent.com/mounicam/lexical_simplification/master/word_complexity_lexicon/lexicon.tsv)
- [SimplePPDB++](https://github.com/mounicam/lexical_simplification/tree/master/SimplePPDBpp) (Use the lexicon version, any size will work)
- [Initial Train Data]()

### For Ranking
- [Initial Train Data]()
- [Initial Test Data]()
- [Language Model (SubIMDB)]()
- [Word2vec Word Embedding]()
- [Google Frequencies]() (Also used for selection)
- [Wikipedia-SimplePPDB Word Ratio]()
- [PPDB Lexicon]()
- [PPDB Phrase Substitutions]()

## Scripts
- `scripts/train_default_cwi.py` - Trains default CWI (see [Setup](## Setup))
- `scripts/train_default_ranker.py` - Trains default ranker (see [Setup](## Setup))
- `scripts/google_freq_compress.py` - Makes Google ngram files smaller by deleting the least frequent ngrams (this is to allow you to use the ngram models on a computer with little memory)
- `scripts/cross_train_cwi_for_paper.py` - Get's a 10-fold average performance of the new Lexi model compared to the old Lexi model, as used in the paper