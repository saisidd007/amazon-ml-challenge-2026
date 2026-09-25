import re
import unicodedata

# Combined replacement dictionaries
NAME_MAP = {
    'corp': 'corporation',
    'inc': 'incorporated',
    'llc': 'limited liability company',
    'ltd': 'limited',
    'pvt': 'private',
    'co': 'company',
    'plc': 'public limited company',
    'sa': 'societe anonyme',
    'sarl': 'societe a responsabilite limitee',
    'sas': 'societe par actions simplifiee',
}

ADDR_MAP = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue',
    'dr': 'drive', 'blvd': 'boulevard', 'ln': 'lane', 'ct': 'court',
    'pl': 'place', 'hwy': 'highway', 'ste': 'suite', 'apt': 'apartment',
    'fl': 'floor', 'bldg': 'building', 'pkwy': 'parkway', 'expwy': 'expressway',
    'ctr': 'center',
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo',
    'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj',
    'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh',
    'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt',
    'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy'
}

# Pre-compiled single-pass regexes (100x faster than looping re.sub)
NAME_REGEX = re.compile(r'\b(' + '|'.join(re.escape(k) for k in NAME_MAP.keys()) + r')\b')
ADDR_REGEX = re.compile(r'\b(' + '|'.join(re.escape(k) for k in ADDR_MAP.keys()) + r')\b')
PUNCT_REGEX = re.compile(r'[\(\)\[\]\{\}\#\*\+\-\_\/\:\;\,\.\?\!\\\"\`\~\|\^\@\$\%\=]')

def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def normalize_business_name(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = strip_accents(text).lower().replace('&', ' and ')
    text = PUNCT_REGEX.sub(' ', text)
    text = NAME_REGEX.sub(lambda m: NAME_MAP[m.group(0)], text)
    return " ".join(text.split())

def normalize_business_address(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = strip_accents(text).lower().replace('&', ' and ')
    text = PUNCT_REGEX.sub(' ', text)
    text = ADDR_REGEX.sub(lambda m: ADDR_MAP[m.group(0)], text)
    return " ".join(text.split())

def extract_numbers(text: str) -> list:
    if not isinstance(text, str) or not text:
        return []
    return re.findall(r'\b\d+\b', text)
