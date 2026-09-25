import re
import unicodedata

# US States mapping
US_STATES = {
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

# Standard word expansions
ADDR_REPLACEMENTS = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bav\b': 'avenue',
    r'\bdr\b': 'drive',
    r'\bblvd\b': 'boulevard',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bhwy\b': 'highway',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bfl\b': 'floor',
    r'\bbldg\b': 'building',
    r'\bpkwy\b': 'parkway',
    r'\bexpwy\b': 'expressway',
    r'\bctr\b': 'center',
    r'\bno\b': '',
    r'\bunit\b': '',
}

NAME_LEGAL_SUFFIXES = {
    r'\bcorp\b': 'corporation',
    r'\binc\b': 'incorporated',
    r'\bllc\b': 'limited liability company',
    r'\bltd\b': 'limited',
    r'\bpvt\b': 'private',
    r'\bco\b': 'company',
    r'\bplc\b': 'public limited company',
    r'\bsa\b': 'societe anonyme',
    r'\bsarl\b': 'societe a responsabilite limitee',
    r'\bsas\b': 'societe par actions simplifiee',
}

def strip_accents(text: str) -> str:
    """Normalize unicode characters and remove diacritics / accents (useful for France/multilingual)."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    return "".join(c for c in text if not unicodedata.combining(c))

def clean_text_basic(text: str) -> str:
    """Basic cleaning, punctuation removal, whitespace normalization."""
    if not isinstance(text, str):
        return ""
    text = strip_accents(text).lower()
    text = text.replace('&', ' and ')
    text = re.sub(r'[\(\)\[\]\{\}\#\*\+\-\_\/\:\;\,\.\?\!\\\"\`\~\|\^\@\$\%\=]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_business_name(name: str) -> str:
    """Cleans business name and standardizes company acronyms."""
    name = clean_text_basic(name)
    for pattern, replacement in NAME_LEGAL_SUFFIXES.items():
        name = re.sub(pattern, replacement, name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_business_address(addr: str) -> str:
    """Cleans address, standardizes street types and US state names."""
    addr = clean_text_basic(addr)
    for state_full, state_abbr in US_STATES.items():
        addr = re.sub(r'\b' + state_full + r'\b', state_abbr, addr)
    for pattern, replacement in ADDR_REPLACEMENTS.items():
        addr = re.sub(pattern, replacement, addr)
    addr = re.sub(r'\s+', ' ', addr).strip()
    return addr

def extract_numbers(text: str) -> list:
    """Extracts all sequence of digits (zip codes, house numbers, pin codes)."""
    if not isinstance(text, str):
        return []
    return re.findall(r'\b\d+\b', text)
