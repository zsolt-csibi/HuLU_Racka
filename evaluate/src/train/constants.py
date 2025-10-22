CONJUNCTIONS = {
    "cause": "mert",
    "effect": "ezért",
}

SST_LABELS = {
    "positive": 1,
    "negative": 2,
    "neutral": 0,
}

CB_LABELS = {
    "entailment": 1,
    "contradiction": 2,
    "neutral": 0,
}

LABELS = {}
LABELS.update(SST_LABELS)
LABELS.update(CB_LABELS)

HULU_DATASETS = {
    "sst": "NYTK/HuSST",
    "rte": "NYTK/HuRTE",
    "wnli": "NYTK/HuWNLI",
    "cola": "NYTK/HuCOLA",
    "copa": "NYTK/HuCoPA",
    "cb": "NYTK/HuCommitmentBank",
}

IRRELEVANT_COLUMNS = {
    "cola": ["id", "sentence"],
    "sst": ["id", "sentence"],
    "wnli": ["id", "sentence1", "sentence2", "orig_id"],
    "rte": ["id", "premise", "hypothesis"],
    "cb": ["id", "premise", "hypothesis"],
    "copa": ["id", "choice1", "choice2", "question"],
}

RELEVANT_COLUMNS = {
    "cola": ["sentence"],
    "sst": ["sentence"],
    "wnli": ["sentence1", "sentence1"],
    "rte": ["premise", "hypothesis"],
    "cb": ["premise", "hypothesis"],
    "copa": ["premise", "choice1", "choice2", "question"],
}

TOKENIZER_PARAMETERS = {
    "cola": {
        "truncation": True,
        "padding": "max_length",
        "max_length": -1,
        "add_special_tokens": True,
    },
    "sst": {
        "truncation": True,
        "padding": "max_length",
        "max_length": -1,
        "add_special_tokens": True,
    },
    "wnli": {
        "truncation": True,
        "padding": "max_length",
        "max_length": -1,
    },
    "rte": {
        "truncation": True,
        "padding": "max_length",
        "max_length": -1,
    },
    "cb": {"truncation": True, "padding": "max_length", "max_length": 512},
    "copa": {"truncation": True, "padding": "max_length", "max_length": 256},
}

CONVERSIONS = {
    "cola": [{"from": "1", "to": "jó"}, {"from": "0", "to": "rossz"}],
    "sst": [{"from": "positive", "to": "pozitív"}, {"from": "negative", "to": "negatív"}, {"from": "neutral", "to": "semleges"}],
    "wnli": [{"from": "1", "to": "igen"}, {"from": "0", "to": "nem"}],
    "rte": [{"from": "1", "to": "igen"}, {"from": "0", "to": "nem"}],
    "cb": [{"from": "entailment", "to": "következik"}, {"from": "contradiction", "to": "ellentétes"}, {"from": "neutral", "to": "semleges"}],
    "copa": [{"from": 0, "to": "első"}, {"from": 1, "to": "második"}],
}

INVERSE_CONVERSIONS = {}
for task, conversions in CONVERSIONS.items():
    INVERSE_CONVERSIONS[task] = [{"from": conv["to"], "to": conv["from"]} for conv in conversions]

COLA_PROMPT = """Az alábbi szöveg a magyar nyelvtani követelményeknek megfelelően jó vagy rossz? 
{examples}
Válaszolj egyetlen szóval: jó vagy rossz.
{text}
Válasz: 
"""

SST_PROMPT = """Az alábbi mondat pozitív, semleges, vagy negatív hangvételű?
{examples}
Válaszolj egyetlen szóval: pozitív, negatív vagy semleges.
{text}
Válasz: 
"""

RTE_PROMPT = """Következik-e a hipotézis a premisszából?
{examples}
premissza: {premise}
hipotézis: {hypothesis}

Válaszolj egyetlen szóval: igen vagy nem.
Válasz: 
"""

CB_PROMPT = """Következik-e a hipotézis a premisszából vagy ellentmond vagy egyik sem? 
{examples}
premissza: {premise}
hipotézis: {hypothesis}
Válaszolj egyetlen szóval: következik, ellentétes vagy semleges.
Válasz: 
"""

COPA_PROMPT_CAUSE = """ A megadott két választás közül melyik az oka a következő eseménynek első vagy második?
Válaszolj egyetlen szóval: első vagy második. Ne adj vissza semmilyen más szöveget.
{examples}
Esemény: {premise}
Első választás: {choice1}
Második választás: {choice2}
Válasz: 
"""

COPA_PROMPT_EFFECT = """ A megadott két választás közül melyik a következő esemény hatása első vagy második?
{examples}
Esemény: {premise}
Első választás: {choice1}
Második választás: {choice2}
Válasz:
"""

WNLI_PROMPT = """A második mondat az elsőnek a következménye-e, igen vagy nem?
{examples}
Első mondat: {sentence1}
Második mondat: {sentence2}
Válaszolj egyetlen szóval: igen vagy nem.
Válasz: 
"""
