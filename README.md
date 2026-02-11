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
From this link: [GOOGLE DRIVE](https://drive.google.com/file/d/1G9RI9qITj3d4fyeJ8e5AXIgEgwjJIq9-/view?usp=sharing)
Save it in the main folder.

### Run it!
```
.venv/bin/python3 SemDeDup.py
```
