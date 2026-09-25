import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict

from preprocessor import normalize_business_name, normalize_business_address
from feature_engineering import compute_pair_features, FEATURE_NAMES

STOPWORDS = {
    'the', 'and', 'or', 'of', 'in', 'at', 'by', 'for', 'with', 'to', 'from',
    'a', 'an', 'is', 'as', 'incorporated', 'corporation', 'limited', 'company',
    'private', 'liability', 'societe', 'anonyme', 'corp', 'inc', 'llc', 'ltd',
    'pvt', 'co', 'plc', 'sa', 'sarl', 'sas'
}

def get_tokens(text: str, min_len: int = 3) -> list:
    if not text:
        return []
    return [tok for tok in text.split() if len(tok) >= min_len and tok not in STOPWORDS]

class UltraFastBlocker:
    """
    Sub-second candidate retrieval using discriminative token posting lists.
    Retrieves candidates ordered by token specificity (lowest posting count).
    """
    def __init__(self, max_candidates_per_s1: int = 35):
        self.max_candidates = max_candidates_per_s1
        
    def generate_candidates(self, s1_df: pd.DataFrame, s23_df: pd.DataFrame) -> dict:
        if len(s1_df) == 0 or len(s23_df) == 0:
            return {eid: [] for eid in s1_df['entity_id']}
            
        print(f"    Indexing {len(s23_df):,} S2/S3 business name tokens...", flush=True)
        token_postings = defaultdict(list)
        s23_names = s23_df['clean_name'].values
        s23_ids = s23_df['entity_id'].values
        
        for idx, name in enumerate(s23_names):
            for tok in get_tokens(name):
                token_postings[tok].append(idx)
                
        print(f"    Retrieving top-{self.max_candidates} candidates for {len(s1_df):,} S1 entities...", flush=True)
        s1_names = s1_df['clean_name'].values
        s1_ids = s1_df['entity_id'].values
        
        result = {}
        for s1_idx in range(len(s1_ids)):
            s1_id = s1_ids[s1_idx]
            s1_name = s1_names[s1_idx]
            toks = get_tokens(s1_name)
            
            if not toks:
                # Fallback to any words
                toks = [w for w in s1_name.split() if len(w) >= 2]
                
            # Sort tokens by rarity (lowest posting length first)
            toks.sort(key=lambda t: len(token_postings.get(t, [])) if token_postings.get(t, []) else 9999999)
            
            cand_indices = []
            seen_indices = set()
            
            for tok in toks:
                postings = token_postings.get(tok, [])
                if postings and len(postings) <= 1500: # skip massive frequency tokens
                    for midx in postings:
                        if midx not in seen_indices:
                            seen_indices.add(midx)
                            cand_indices.append(midx)
                            if len(cand_indices) >= self.max_candidates:
                                break
                if len(cand_indices) >= self.max_candidates:
                    break
                    
            result[s1_id] = [s23_ids[idx] for idx in cand_indices]
            
        return result

def main():
    print("=" * 70, flush=True)
    print("  BUSINESS ENTITY RESOLUTION — HIGH-SPEED PRODUCTION PIPELINE", flush=True)
    print("=" * 70, flush=True)
    
    t_start = time.time()
    train_dir = '6ab10eb3b23ba_student_resource/student_resource/dataset/train'
    test_dir = '6ab10eb3b23ba_student_resource/student_resource/dataset/test'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # STAGE 1: LOAD GROUND TRUTH & TRAIN REGULARIZED MODEL
    # -------------------------------------------------------------
    print("\n>>> STAGE 1: MODEL TRAINING & CALIBRATION <<<", flush=True)
    gt_df = pd.read_csv(os.path.join(train_dir, 'train_ground_truth.tsv'), sep='\t', engine='pyarrow')
    gt_dict = {}
    for s1_id, mids in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        if pd.notna(mids) and str(mids).strip():
            gt_dict[s1_id] = set(str(mids).strip().split(','))
        else:
            gt_dict[s1_id] = set()
            
    s1_train_full = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t', engine='pyarrow')
    s2_train_full = pd.read_csv(os.path.join(train_dir, 'train_source2.tsv'), sep='\t', engine='pyarrow')
    s3_train_full = pd.read_csv(os.path.join(train_dir, 'train_source3.tsv'), sep='\t', engine='pyarrow')
    
    # 25,000 stratified training entities
    train_sample_size = 25000
    s1_us = s1_train_full[s1_train_full['country'] == 'US'].sample(n=train_sample_size // 2, random_state=42)
    s1_in = s1_train_full[s1_train_full['country'] == 'India'].sample(n=train_sample_size // 2, random_state=42)
    s1_train_sample = pd.concat([s1_us, s1_in]).reset_index(drop=True)
    
    needed_train_ids = set()
    for s1_id in s1_train_sample['entity_id']:
        needed_train_ids.update(gt_dict.get(s1_id, set()))
        
    s23_train_pool = pd.concat([
        s2_train_full[s2_train_full['entity_id'].isin(needed_train_ids)],
        s2_train_full.sample(n=80000, random_state=42),
        s3_train_full[s3_train_full['entity_id'].isin(needed_train_ids)],
        s3_train_full.sample(n=80000, random_state=42)
    ]).drop_duplicates('entity_id').reset_index(drop=True)
    
    print("  Preprocessing training entities...", flush=True)
    s1_train_sample['clean_name'] = s1_train_sample['business_name'].fillna('').apply(normalize_business_name)
    s1_train_sample['clean_address'] = s1_train_sample['business_address'].fillna('').apply(normalize_business_address)
    
    s23_train_pool['clean_name'] = s23_train_pool['business_name'].fillna('').apply(normalize_business_name)
    s23_train_pool['clean_address'] = s23_train_pool['business_address'].fillna('').apply(normalize_business_address)
    s23_train_dict = s23_train_pool.set_index('entity_id').to_dict('index')
    
    blocker = UltraFastBlocker(max_candidates_per_s1=35)
    train_cands = {}
    for country in s1_train_sample['country'].unique():
        cands = blocker.generate_candidates(
            s1_train_sample[s1_train_sample['country'] == country],
            s23_train_pool[s23_train_pool['country'] == country]
        )
        train_cands.update(cands)
        
    print("  Extracting dense similarity features for training...", flush=True)
    X_train, y_train = [], []
    s1_train_dict = s1_train_sample.set_index('entity_id').to_dict('index')
    
    for s1_id, cands in train_cands.items():
        s1_info = s1_train_dict[s1_id]
        true_mids = gt_dict.get(s1_id, set())
        
        for cand_id in cands:
            if cand_id in s23_train_dict:
                cand_info = s23_train_dict[cand_id]
                feats = compute_pair_features(
                    s1_info['clean_name'], s1_info['clean_address'],
                    cand_info['clean_name'], cand_info['clean_address'],
                    cand_id
                )
                X_train.append(feats)
                y_train.append(1 if cand_id in true_mids else 0)
                
        for true_id in true_mids:
            if true_id in s23_train_dict and true_id not in cands:
                cand_info = s23_train_dict[true_id]
                feats = compute_pair_features(
                    s1_info['clean_name'], s1_info['clean_address'],
                    cand_info['clean_name'], cand_info['clean_address'],
                    true_id
                )
                X_train.append(feats)
                y_train.append(1)
                
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"  Training Set: {len(X_train):,} pairs (Positives: {np.sum(y_train):,}, Negatives: {len(y_train)-np.sum(y_train):,})", flush=True)
    
    print("  Fitting Regularized LightGBM Classifier...", flush=True)
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'feature_fraction': 0.80,
        'bagging_fraction': 0.80,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_data_in_leaf': 30,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
    model = lgb.train(params, lgb_train, num_boost_round=300)
    
    del s1_train_full, s2_train_full, s3_train_full, s23_train_pool, X_train, y_train, gt_df
    
    # -------------------------------------------------------------
    # STAGE 2: STREAMING TEST INFERENCE BY COUNTRY
    # -------------------------------------------------------------
    print("\n>>> STAGE 2: STREAMING TEST SET INFERENCE <<<", flush=True)
    optimal_threshold = 0.65
    
    print("  Loading Test Datasets with PyArrow...", flush=True)
    test_s1 = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t', engine='pyarrow')
    test_s2 = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t', engine='pyarrow')
    test_s3 = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t', engine='pyarrow')
    
    test_s23 = pd.concat([test_s2, test_s3]).reset_index(drop=True)
    del test_s2, test_s3
    
    print(f"  Test Source 1: {len(test_s1):,} rows", flush=True)
    print(f"  Test Source 2+3: {len(test_s23):,} rows", flush=True)
    print(f"  Test Countries: {test_s1['country'].value_counts().to_dict()}", flush=True)
    
    matching_results_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_pairs_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    
    with open(matching_results_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_pairs_path, 'w', encoding='utf-8') as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
    for country in test_s1['country'].unique():
        print(f"\n  --- Processing Country: {country} ---", flush=True)
        t_c = time.time()
        c_s1 = test_s1[test_s1['country'] == country].copy().reset_index(drop=True)
        c_s23 = test_s23[test_s23['country'] == country].copy().reset_index(drop=True)
        print(f"    Entities: {len(c_s1):,} S1 vs {len(c_s23):,} S2/S3", flush=True)
        
        c_s1['clean_name'] = c_s1['business_name'].fillna('').apply(normalize_business_name)
        c_s1['clean_address'] = c_s1['business_address'].fillna('').apply(normalize_business_address)
        
        c_s23['clean_name'] = c_s23['business_name'].fillna('').apply(normalize_business_name)
        c_s23['clean_address'] = c_s23['business_address'].fillna('').apply(normalize_business_address)
        
        c_s23_dict = c_s23.set_index('entity_id').to_dict('index')
        c_s1_dict = c_s1.set_index('entity_id').to_dict('index')
        
        # Fast Inverted Index Candidate Generation
        cands_dict = blocker.generate_candidates(c_s1, c_s23)
        
        print("    Scoring candidates with LightGBM in streaming chunks...", flush=True)
        s1_ids_country = c_s1['entity_id'].values
        
        chunk_size = 30000
        with open(matching_results_path, 'a', encoding='utf-8') as f_match, \
             open(candidate_pairs_path, 'a', encoding='utf-8') as f_cand:
            
            for c_start in range(0, len(s1_ids_country), chunk_size):
                c_end = min(c_start + chunk_size, len(s1_ids_country))
                batch_s1_ids = s1_ids_country[c_start:c_end]
                
                batch_pairs = []
                batch_metadata = []
                
                for s1_id in batch_s1_ids:
                    s1_info = c_s1_dict[s1_id]
                    cands = cands_dict.get(s1_id, [])
                    for cand_id in cands:
                        if cand_id in c_s23_dict:
                            cand_info = c_s23_dict[cand_id]
                            feats = compute_pair_features(
                                s1_info['clean_name'], s1_info['clean_address'],
                                cand_info['clean_name'], cand_info['clean_address'],
                                cand_id
                            )
                            batch_pairs.append(feats)
                            batch_metadata.append((s1_id, cand_id))
                            
                s1_to_matches = defaultdict(list)
                if batch_pairs:
                    X_eval = np.array(batch_pairs)
                    scores = model.predict(X_eval)
                    for (s1_id, cand_id), score in zip(batch_metadata, scores):
                        if score >= optimal_threshold:
                            s1_to_matches[s1_id].append(cand_id)
                            
                for s1_id in batch_s1_ids:
                    cands_list = cands_dict.get(s1_id, [])
                    matched_list = s1_to_matches.get(s1_id, [])
                    
                    cand_str = ",".join(cands_list)
                    match_str = ",".join(matched_list)
                    
                    f_cand.write(f"{s1_id}\t{cand_str}\n")
                    f_match.write(f"{s1_id}\t{match_str}\n")
                    
                print(f"      Streamed {c_end:,} / {len(s1_ids_country):,} entities...", flush=True)
                
        print(f"    Completed Country {country} in {time.time()-t_c:.2f}s", flush=True)
        del c_s1, c_s23, c_s23_dict, c_s1_dict, cands_dict
        
    print("\n" + "=" * 70, flush=True)
    print(f"  ALL TEST INFERENCE COMPLETED IN {time.time()-t_start:.2f} SECONDS!", flush=True)
    print("=" * 70, flush=True)

if __name__ == '__main__':
    main()
