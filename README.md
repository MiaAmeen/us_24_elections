# Hey Andy !

## Installation

### 1. First: clone the repo, obviously.
```
git clone https://github.com/MiaAmeen/us_24_elections.git
cd us_24_elections
```
### Then, start python venv and install required packages:
...Make sure you have python >= 3.10. Use your python alias.
```
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python3 -m pip install -r requirements.txt
```
### Additionally, copy/paste required credentials to the .env file.
```
.env_copy --> .env
```
### Download the data:
Download the files from these links: 
- [DATASET #1](https://drive.google.com/file/d/1iai2XHZAEot-7jr2bqPKwJc7akoVbvKs/view?usp=sharing)
- [DATASET #2](https://drive.google.com/file/d/1txieLz4LLIqQwMq28ln166Z05saA3pPI/view?usp=sharing)

And make sure to store them in the us_24_elections/data/ folder:
```
mkdir data
```

### Run it!
```
.venv/bin/python3 SemDeDup.py
```
