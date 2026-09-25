import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

STOPWORDS = {
    'the', 'and', 'or', 'of', 'in', 'at', 'by', 'for', 'with', 'to', 'from',
    'a', 'an', 'is', 'as', 'incorporated', 'corporation', 'limited', 'company',
    'private', 'liability', 'societe', 'anonyme'
}

def get_significant_tokens(text: str, min_len: int = 3) -> set:
    tokens = set()
    if not text:
        return tokens
    for tok in text.split():
        if len(tok) >= min_len and tok not in STOPWORDS:
            tokens.add(tok)
    return tokens

class HighRecallCountryBlocker:
    """
    High-Recall, High-Speed Multi-Channel Blocker.
    Uses Char 3-4 gram TF-IDF (robust to typos) + Word Token Inverted Index.
    """
    def __init__(self, top_k_tfidf: int = 35, max_candidates_per_s1: int = 50):
        self.top_k_tfidf = top_k_tfidf
        self.max_candidates = max_candidates_per_s1
        
    def generate_candidates_for_country(
        self,
        s1_df: pd.DataFrame,
        s23_df: pd.DataFrame
    ) -> dict:
        if len(s1_df) == 0 or len(s23_df) == 0:
            return {eid: [] for eid in s1_df['entity_id']}
            
        candidates_map = defaultdict(set)
        s1_ids = s1_df['entity_id'].values
        s23_ids = s23_df['entity_id'].values
        
        # 1. Channel A: Name Token Inverted Index
        print("    Building Name Token Inverted Index...", flush=True)
        name_token_index = defaultdict(list)
        for idx, name in enumerate(s23_df['clean_name']):
            for tok in get_significant_tokens(name):
                name_token_index[tok].append(idx)
                
        # 2. Channel B: Char 3-4 gram TF-IDF on Combined Text
        print("    Fitting Char (3,4)-gram TF-IDF Vectorizer...", flush=True)
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 4),
            min_df=3,
            max_df=0.98,
            max_features=35000,
            sublinear_tf=True
        )
        
        tfidf_s23 = vectorizer.fit_transform(s23_df['full_text'])
        tfidf_s1 = vectorizer.transform(s1_df['full_text'])
        
        print(f"    Computing sparse dot product for {len(s1_df):,} S1 vs {len(s23_df):,} S2/S3...", flush=True)
        batch_size = 5000
        n_s1 = len(s1_df)
        
        for start_idx in range(0, n_s1, batch_size):
            end_idx = min(start_idx + batch_size, n_s1)
            batch_s1 = tfidf_s1[start_idx:end_idx]
            
            sim_matrix = batch_s1.dot(tfidf_s23.T)
            
            for local_row_idx in range(sim_matrix.shape[0]):
                global_s1_idx = start_idx + local_row_idx
                s1_id = s1_ids[global_s1_idx]
                
                row_start = sim_matrix.indptr[local_row_idx]
                row_end = sim_matrix.indptr[local_row_idx + 1]
                
                if row_end > row_start:
                    indices = sim_matrix.indices[row_start:row_end]
                    data = sim_matrix.data[row_start:row_end]
                    
                    if len(data) > self.top_k_tfidf:
                        top_k_sub = np.argpartition(data, -self.top_k_tfidf)[-self.top_k_tfidf:]
                        top_s23_indices = indices[top_k_sub]
                    else:
                        top_s23_indices = indices
                        
                    for s23_idx in top_s23_indices:
                        candidates_map[s1_id].add(s23_ids[s23_idx])
                        
                # Name Token Inverted Index Matches
                s1_name = s1_df['clean_name'].values[global_s1_idx]
                s1_tokens = get_significant_tokens(s1_name)
                for tok in s1_tokens:
                    matched_idxs = name_token_index.get(tok, [])
                    if 0 < len(matched_idxs) <= 300:
                        for midx in matched_idxs[:25]:
                            candidates_map[s1_id].add(s23_ids[midx])
                            
        result = {}
        for s1_id in s1_ids:
            cands = list(candidates_map.get(s1_id, []))
            if len(cands) > self.max_candidates:
                cands = cands[:self.max_candidates]
            result[s1_id] = cands
            
        return result
