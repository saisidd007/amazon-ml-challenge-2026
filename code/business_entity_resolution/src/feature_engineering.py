import re
import numpy as np
from rapidfuzz import fuzz, distance
import jellyfish

def get_tokens(text: str) -> set:
    if not text:
        return set()
    return set(text.split())

def extract_digit_tokens(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r'\b\d+\b', text))

def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    cand_name: str,
    cand_addr: str,
    cand_id: str
) -> list:
    """
    Computes a vector of dense similarity and distance features for a candidate pair.
    """
    # 1. Name features
    s1_name = s1_name or ""
    cand_name = cand_name or ""
    
    name_lev = distance.Levenshtein.normalized_similarity(s1_name, cand_name)
    name_jw = distance.JaroWinkler.similarity(s1_name, cand_name)
    name_sort_ratio = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    name_set_ratio = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    name_partial_ratio = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    
    t1_name = get_tokens(s1_name)
    t2_name = get_tokens(cand_name)
    inter_name = len(t1_name & t2_name)
    union_name = len(t1_name | t2_name)
    name_jaccard = inter_name / union_name if union_name > 0 else 0.0
    min_tokens_name = min(len(t1_name), len(t2_name)) if min(len(t1_name), len(t2_name)) > 0 else 1
    name_containment = inter_name / min_tokens_name
    
    name_len_diff = abs(len(s1_name) - len(cand_name))
    name_len_ratio = (min(len(s1_name), len(cand_name)) / max(len(s1_name), len(cand_name))) if max(len(s1_name), len(cand_name)) > 0 else 1.0
    
    exact_name = 1.0 if s1_name == cand_name and len(s1_name) > 0 else 0.0
    first_char_name = 1.0 if (s1_name and cand_name and s1_name[0] == cand_name[0]) else 0.0
    
    s1_first_word = s1_name.split()[0] if s1_name else ""
    cand_first_word = cand_name.split()[0] if cand_name else ""
    first_word_match = 1.0 if (s1_first_word and s1_first_word == cand_first_word) else 0.0
    
    # Phonetic Soundex match on first word
    try:
        soundex_match = 1.0 if (s1_first_word and cand_first_word and jellyfish.soundex(s1_first_word) == jellyfish.soundex(cand_first_word)) else 0.0
    except Exception:
        soundex_match = 0.0

    # 2. Address features
    s1_addr = s1_addr or ""
    cand_addr = cand_addr or ""
    
    addr_lev = distance.Levenshtein.normalized_similarity(s1_addr, cand_addr)
    addr_jw = distance.JaroWinkler.similarity(s1_addr, cand_addr)
    addr_sort_ratio = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
    addr_set_ratio = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
    addr_partial_ratio = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
    
    t1_addr = get_tokens(s1_addr)
    t2_addr = get_tokens(cand_addr)
    inter_addr = len(t1_addr & t2_addr)
    union_addr = len(t1_addr | t2_addr)
    addr_jaccard = inter_addr / union_addr if union_addr > 0 else 0.0
    min_tokens_addr = min(len(t1_addr), len(t2_addr)) if min(len(t1_addr), len(t2_addr)) > 0 else 1
    addr_containment = inter_addr / min_tokens_addr
    
    addr_len_diff = abs(len(s1_addr) - len(cand_addr))
    exact_addr = 1.0 if s1_addr == cand_addr and len(s1_addr) > 0 else 0.0
    
    # 3. Numeric / Postal / House Number Overlap
    num1 = extract_digit_tokens(s1_addr)
    num2 = extract_digit_tokens(cand_addr)
    num_inter = len(num1 & num2)
    num_union = len(num1 | num2)
    num_jaccard = num_inter / num_union if num_union > 0 else (1.0 if len(num1) == 0 and len(num2) == 0 else 0.0)
    has_num_mismatch = 1.0 if (len(num1) > 0 and len(num2) > 0 and num_inter == 0) else 0.0
    
    # 4. Source prefix (S2 vs S3)
    is_source2 = 1.0 if cand_id.startswith('S2-') else 0.0
    is_source3 = 1.0 if cand_id.startswith('S3-') else 0.0
    
    # 5. Combined interactions
    name_x_addr_jw = name_jw * addr_jw
    name_x_addr_set = name_set_ratio * addr_set_ratio
    overall_avg_sim = (name_jw + name_set_ratio + addr_jw + addr_set_ratio) / 4.0
    
    return [
        name_lev,
        name_jw,
        name_sort_ratio,
        name_set_ratio,
        name_partial_ratio,
        name_jaccard,
        name_containment,
        name_len_diff,
        name_len_ratio,
        exact_name,
        first_char_name,
        first_word_match,
        soundex_match,
        addr_lev,
        addr_jw,
        addr_sort_ratio,
        addr_set_ratio,
        addr_partial_ratio,
        addr_jaccard,
        addr_containment,
        addr_len_diff,
        exact_addr,
        num_inter,
        num_jaccard,
        has_num_mismatch,
        is_source2,
        is_source3,
        name_x_addr_jw,
        name_x_addr_set,
        overall_avg_sim
    ]

FEATURE_NAMES = [
    'name_lev', 'name_jw', 'name_sort_ratio', 'name_set_ratio', 'name_partial_ratio',
    'name_jaccard', 'name_containment', 'name_len_diff', 'name_len_ratio', 'exact_name',
    'first_char_name', 'first_word_match', 'soundex_match', 'addr_lev', 'addr_jw',
    'addr_sort_ratio', 'addr_set_ratio', 'addr_partial_ratio', 'addr_jaccard', 'addr_containment',
    'addr_len_diff', 'exact_addr', 'num_inter', 'num_jaccard', 'has_num_mismatch',
    'is_source2', 'is_source3', 'name_x_addr_jw', 'name_x_addr_set', 'overall_avg_sim'
]
