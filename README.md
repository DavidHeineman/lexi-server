# Lexi-English

Implements a backend for an English version of [Lexi](https://www.aclweb.org/anthology/C18-1021.pdf). Uses a different CWI and ranking model as described [in this paper](https://www.aclweb.org/anthology/D18-1410.pdf).

## Setup

- Download data to `res/en` (see below)
- Install and setup Postgres database using `lexi_schema.pgsql`
- Install requirements: `pip install -r requirements.txt`
- Train the default readability ranker with `python scripts/train_default_ranker.py`
- Run the server with `python lexi/server/run_lexi_server.py -S`
- Install [Lexi Frontend](https://github.com/jbingel/lexi-frontend) and it will work in English.

**Note:** If you're not running Lexi on `localhost`, use `lexi.cfg` to modify the server configuration.

### Modifying CWI/Ranking Configuration

See `lexi/config.py` for changing project-wide variables:
- `DEFAULT_THRESHOLD`   - The default threshold for the CWI to declare a word "complex"
- `NUM_REPLACEMENTS`    - The number of replacements generated from the PPDB++ lexicon

## Data

Data is located in the `res/en` folder. Modify `lexi/config.py` to point the project to your data.

### For CWI

- [Complexity Lexicon](https://raw.githubusercontent.com/mounicam/lexical_simplification/master/word_complexity_lexicon/lexicon.tsv)
- [SimplePPDB++](https://github.com/mounicam/lexical_simplification/tree/master/SimplePPDBpp) (Use the lexicon version, any size will work)

### For Ranking
- [Initial Train Data]()
- [Initial Test Data]()
- [Language Model (SubIMDB)]()
- [Word2vec Word Embedding]()
- [Google Frequencies]()
- [Wikipedia-SimplePPDB Word Ratio]()
- [PPDB Scores]()

### Note: Differences between Lexi and Lexi-English
- Lexi's CWI model is trained on user data, while Lexi-English uses a simple complexity lexicon. In Lexi English, each user's threshold for what counts as a complex word is different rather than the CWI model itself.
- The substitution generator on Lexi got either thesaurus or the closest word2vec words, while Lexi-English generates candidates from SimplePPDB++. Candidates under a certian threshold for similarity are removed and are initally fed into the ranker by their complexity scores as determined by the dataset.