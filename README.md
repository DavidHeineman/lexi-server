# Lexi-English

Implements a backend for an English version of [Lexi](https://www.aclweb.org/anthology/C18-1021.pdf). Uses a different CWI and ranking model as described [in this paper](https://www.aclweb.org/anthology/D18-1410.pdf).

## Data

Data is located in the `/res/en` folder. Modify `/lexi/config.py` to point the project to your data.

## Setup

Install and setup Postgres database

`pip install -r requirements.txt`

`python run_lexi_server.py -S`

Install [Lexi Frontend](https://github.com/jbingel/lexi-frontend) and it will work in English.

**Note:** If you're not running Lexi on `localhost`, use `lexi.cfg` to modify the server configuration.

## Differences between Lexi and Lexi-English