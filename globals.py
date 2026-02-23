from pathlib import Path


# --------- FILE PATHS ----------
TRUTH_SOCIAL_FILE = "./TS_parquets/TS24_cleaned.parquet"
BLUESKY_FILE      = "./bsky_scrape/BS24.parquet"
TWITTER_FOLDER = "./X_parquets_clean"  # contains multiple .parquet
TWITTER_FILES = list(Path(TWITTER_FOLDER).glob("*.parquet"))


# --------- OTHER variables ----------
CREATED_AT_COL = "created_at"
ID_COL = "external_id"
CLEAN_TEXT_COL = "clean_content"

MONTHS = ["2024-05","2024-06","2024-07","2024-08","2024-09","2024-10","2024-11"]
month_map = {"may": "05", "jun": "06", "jul": "07", "aug": "08", "sept": "09", "oct": "10", "nov": "11"}


# ------- CLEANED keywords ------------ (cleaned using clean_keywords)
X_KEYWORDS = ['2024 Elections', '2024 Presidential Election', 'Biden', 'Biden2024', 'CPAC', 'Cornel West', 'DNC', 'Dean Phillips', 'Democratic party', 'Donald Trump', 'GOP', 'Green Party', 'Independent Party', 'Jill Stein', 'Joe Biden', 'Joe Biden and Kamala Harris', 'Joseph Biden', 'KAG', 'Kamala Harris', 'MAGA', 'Marianne Williamson', 'Nikki Haley', 'No Labels', 'RFK Jr', 'RNC', 'Republican party', 'Robert F. Kennedy Jr.', 'Ron DeSantis', 'Snowballing', 'Third Party', 'Trump2024', 'US Elections', 'Vivek Ramaswamy', 'bidenharris2024', 'conservative', 'letsgobrandon', 'makeamericagreatagain', 'phillips2024', 'thedemocrats', 'trumpsupporters', 'trumptrain', 'ultramaga', 'voteblue2024', 'williamson2024']
TS_KEYWORDS = ['2024Elections', '2024PresidentialElections', '2024USElections', 'Biden', 'Biden2024', 'CPAC', 'CornellWest', 'DNC', 'Democraticparty', 'DonaldTrump', 'GOP', 'GreenParty', 'IndependentParty', 'JillStein', 'JoeBiden', 'JosephBiden', 'KAG', 'KamalaHarris', 'MAGA', 'MarianneWilliamson', 'DeanPhillips', 'NikkiHaley', 'NoLabels', 'RFKJr', 'RNC', 'Republicanparty', 'RobertF.KennedyJr.', 'RonDeSantis', 'ThirdParty', 'Trump2024', 'USElections', 'VivekRamaswamy', 'bidenharris2024', 'conservative', 'democratsoftiktok', 'letsgobrandon', 'makeamericagreatagain', 'phillips2024', 'republicansoftiktok', 'thedemocrats', 'trumpsupporters', 'trumptrain', 'ultramaga', 'voteblue2024', 'williamson2024']
# somewhat union of keywords used for data collection (look at clean_keywords output)
KEYWORDS = ['2024elections', '2024presidentialelection', '2024presidentialelections', 'biden', 'biden2024', 'bidenharris2024', 'conservative', 'cpac', 'cornelwest', 'joebidenandkamalaharris', 'deanphillips', 'democraticparty', 'dnc', 'donaldtrump', 'gop', 'greenparty', 'independentparty', 'jillstein', 'joebiden', 'josephbiden', 'kag', 'kamalaharris', 'letsgobrandon', 'maga', 'makeamericagreatagain', 'mariannewilliamson', 'nikkihaley', 'nolabels', 'phillips2024', 'republicanparty', 'rfkjr', 'rnc', 'robertf.kennedyjr.', 'rondesantis', 'thedemocrats', 'thirdparty', 'trump2024', 'trumpsupporters', 'trumptrain', 'ultramaga', 'uselections', 'vivekramaswamy', 'voteblue2024', 'williamson2024']

