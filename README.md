# Lexi-English

Implements a backend for an English version of [Lexi](https://www.aclweb.org/anthology/C18-1021.pdf). Uses a different CWI and ranking model as described [in this paper](https://www.aclweb.org/anthology/D18-1410.pdf).

## Setup

- Download data to `/res/en` (see below)
- Install and setup Postgres database using `/lexi_schema.pgsql`
- Install requirements: `pip install -r requirements.txt`
- Train the default readability ranker with `python scripts/train_default_ranker.py`
- Run the server with `python /lexi/server/run_lexi_server.py -S`
- Install [Lexi Frontend](https://github.com/jbingel/lexi-frontend) and it will work in English.

**Note:** If you're not running Lexi on `localhost`, use `lexi.cfg` to modify the server configuration.

## Data

Data is located in the `/res/en` folder. Modify `/lexi/config.py` to point the project to your data.

[Download Complexity Lexicon](https://raw.githubusercontent.com/mounicam/lexical_simplification/master/word_complexity_lexicon/lexicon.tsv)

[Download SimplePPDB++](https://github.com/mounicam/lexical_simplification/tree/master/SimplePPDBpp) (Any size will work)

### Note: Differences between Lexi and Lexi-English
- Lexi's CWI model is trained on user data, while Lexi-English uses a simple complexity lexicon. In Lexi English, each user's threshold for what counts as a complex word is different rather than the CWI model itself.