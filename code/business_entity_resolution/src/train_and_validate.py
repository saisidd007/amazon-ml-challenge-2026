import os
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict
from sklearn.model_selection import train_test_split
from preprocessor import normalize_business_name, normalize_business_address
from blocking import HighRecallCountryBlocker as CountryBlocker
from feature_engineering import compute_pair_features, FEATURE_NAMES
from metrics import evaluate_macro_f05, compute_blocking_recall

def run_experiment():
    print("=" * 60, flush=True)
    print("  BUSINESS ENTITY RESOLUTION: FAST TRAINING & VALIDATION", flush=True)
    print("=" * 60, flush=True)
    
    t0 = time.time()
    train_dir = '6ab10eb3b23ba_student_resource/student_resource/dataset/train'
    
    # 1. Load Ground Truth using PyArrow
    print("Loading Ground Truth with pyarrow engine...", flush=True)
    gt_df = pd.read_csv(os.path.join(train_dir, 'train_ground_truth.tsv'), sep='\t', engine='pyarrow')
    gt_dict = {}
    for s1_id, mids in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        if pd.notna(mids) and str(mids).strip():
            gt_dict[s1_id] = set(str(mids).strip().split(','))
        else:
            gt_dict[s1_id] = set()
            
    print(f"Loaded {len(gt_dict):,} Ground Truth Entities in {time.time()-t0:.2f}s", flush=True)
    
    # 2. Load Source 1
    t1 = time.time()
    print("Loading Source 1...", flush=True)
    s1_all = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t', engine='pyarrow')
    print(f"Loaded Source 1 ({len(s1_all):,} rows) in {time.time()-t1:.2f}s", flush=True)
    
    # Sample stratified split
    val_sample_size = 20000
    s1_us = s1_all[s1_all['country'] == 'US'].sample(n=val_sample_size // 2, random_state=42)
    s1_in = s1_all[s1_all['country'] == 'India'].sample(n=val_sample_size // 2, random_state=42)
    s1_sampled = pd.concat([s1_us, s1_in]).reset_index(drop=True)
    
    train_s1_df, val_s1_df = train_test_split(
        s1_sampled, test_size=0.33, random_state=42, stratify=s1_sampled['country']
    )
    
    print(f"Split: {len(train_s1_df):,} Train S1 entities, {len(val_s1_df):,} Validation S1 entities.", flush=True)
    
    # Preprocess S1
    print("Preprocessing Source 1 entities...", flush=True)
    train_s1_df = train_s1_df.copy()
    val_s1_df = val_s1_df.copy()
    for df in [train_s1_df, val_s1_df]:
        df['clean_name'] = df['business_name'].fillna('').apply(normalize_business_name)
        df['clean_address'] = df['business_address'].fillna('').apply(normalize_business_address)
        df['full_text'] = df['clean_name'] + ' ' + df['clean_address']
        
    # 3. Load Source 2 & Source 3 with PyArrow
    t2 = time.time()
    print("Loading Source 2 and Source 3 with PyArrow...", flush=True)
    s2_all = pd.read_csv(os.path.join(train_dir, 'train_source2.tsv'), sep='\t', engine='pyarrow')
    s3_all = pd.read_csv(os.path.join(train_dir, 'train_source3.tsv'), sep='\t', engine='pyarrow')
    print(f"Loaded Source 2 ({len(s2_all):,} rows) & Source 3 ({len(s3_all):,} rows) in {time.time()-t2:.2f}s", flush=True)
    
    # Collect all needed match IDs for sampled S1
    needed_match_ids = set()
    for s1_id in s1_sampled['entity_id']:
        needed_match_ids.update(gt_dict.get(s1_id, set()))
        
    s2_needed = s2_all[s2_all['entity_id'].isin(needed_match_ids)]
    s2_bg = s2_all.sample(n=100000, random_state=42)
    s2_sub = pd.concat([s2_needed, s2_bg]).drop_duplicates('entity_id')
    
    s3_needed = s3_all[s3_all['entity_id'].isin(needed_match_ids)]
    s3_bg = s3_all.sample(n=100000, random_state=42)
    s3_sub = pd.concat([s3_needed, s3_bg]).drop_duplicates('entity_id')
    
    s23_all = pd.concat([s2_sub, s3_sub]).drop_duplicates('entity_id').reset_index(drop=True)
    print(f"Candidate Pool: {len(s23_all):,} entities across S2 & S3.", flush=True)
    
    # Clean S23
    print("Preprocessing Candidate pool...", flush=True)
    s23_all['clean_name'] = s23_all['business_name'].fillna('').apply(normalize_business_name)
    s23_all['clean_address'] = s23_all['business_address'].fillna('').apply(normalize_business_address)
    s23_all['full_text'] = s23_all['clean_name'] + ' ' + s23_all['clean_address']
    
    s23_dict = s23_all.set_index('entity_id').to_dict('index')
    
    # 4. Candidate Generation (Blocking)
    print("\n--- STEP 1: CANDIDATE GENERATION (BLOCKING) ---", flush=True)
    blocker = CountryBlocker(top_k_tfidf=20, max_candidates_per_s1=30)
    
    train_candidates = {}
    for country in train_s1_df['country'].unique():
        print(f"  Generating Train candidates for country: {country}...", flush=True)
        s1_sub = train_s1_df[train_s1_df['country'] == country]
        s23_sub = s23_all[s23_all['country'] == country]
        cands = blocker.generate_candidates_for_country(s1_sub, s23_sub)
        train_candidates.update(cands)
        
    val_candidates = {}
    for country in val_s1_df['country'].unique():
        print(f"  Generating Val candidates for country: {country}...", flush=True)
        s1_sub = val_s1_df[val_s1_df['country'] == country]
        s23_sub = s23_all[s23_all['country'] == country]
        cands = blocker.generate_candidates_for_country(s1_sub, s23_sub)
        val_candidates.update(cands)
        
    val_gt = {eid: gt_dict.get(eid, set()) for eid in val_s1_df['entity_id']}
    overall_rec, macro_rec = compute_blocking_recall(val_gt, val_candidates)
    print(f"\n>>> Validation Blocking Overall Recall: {overall_rec * 100:.2f}% <<<", flush=True)
    print(f">>> Validation Blocking Macro Recall:   {macro_rec * 100:.2f}% <<<", flush=True)
    
    # 5. Feature Extraction for Training
    print("\n--- STEP 2: FEATURE EXTRACTION FOR TRAINING ---", flush=True)
    t_feat = time.time()
    X_train, y_train = [], []
    train_s1_dict = train_s1_df.set_index('entity_id').to_dict('index')
    
    for s1_id, cands in train_candidates.items():
        s1_info = train_s1_dict[s1_id]
        true_mids = gt_dict.get(s1_id, set())
        
        for cand_id in cands:
            if cand_id in s23_dict:
                cand_info = s23_dict[cand_id]
                feats = compute_pair_features(
                    s1_info['clean_name'],
                    s1_info['clean_address'],
                    cand_info['clean_name'],
                    cand_info['clean_address'],
                    cand_id
                )
                is_match = 1 if cand_id in true_mids else 0
                X_train.append(feats)
                y_train.append(is_match)
                
        for true_id in true_mids:
            if true_id in s23_dict and true_id not in cands:
                cand_info = s23_dict[true_id]
                feats = compute_pair_features(
                    s1_info['clean_name'],
                    s1_info['clean_address'],
                    cand_info['clean_name'],
                    cand_info['clean_address'],
                    true_id
                )
                X_train.append(feats)
                y_train.append(1)
                
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Extracted features for {len(X_train):,} pairs in {time.time()-t_feat:.2f}s (Positives: {np.sum(y_train):,}, Negatives: {len(y_train)-np.sum(y_train):,})", flush=True)
    
    # 6. Train LightGBM Classifier
    print("\n--- STEP 3: TRAINING LIGHTGBM CLASSIFIER ---", flush=True)
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
    
    model = lgb.train(params, lgb_train, num_boost_round=300)
    
    os.makedirs('code/business_entity_resolution/src/models', exist_ok=True)
    with open('code/business_entity_resolution/src/models/lgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to code/business_entity_resolution/src/models/lgb_model.pkl", flush=True)
    
    # Feature Importance
    importance = model.feature_importance(importance_type='gain')
    top_features = sorted(zip(FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Most Important Features:", flush=True)
    for fname, imp in top_features:
        print(f"  {fname:25s}: {imp:,.2f}", flush=True)
        
    # 7. Validation Inference & Threshold Tuning
    print("\n--- STEP 4: VALIDATION & F_0.5 THRESHOLD TUNING ---", flush=True)
    val_s1_dict = val_s1_df.set_index('entity_id').to_dict('index')
    
    val_pairs_features = []
    val_pair_metadata = []
    
    for s1_id, cands in val_candidates.items():
        s1_info = val_s1_dict[s1_id]
        for cand_id in cands:
            if cand_id in s23_dict:
                cand_info = s23_dict[cand_id]
                feats = compute_pair_features(
                    s1_info['clean_name'],
                    s1_info['clean_address'],
                    cand_info['clean_name'],
                    cand_info['clean_address'],
                    cand_id
                )
                val_pairs_features.append(feats)
                val_pair_metadata.append((s1_id, cand_id))
                
    val_X = np.array(val_pairs_features)
    val_scores = model.predict(val_X)
    
    s1_to_scored_cands = defaultdict(list)
    for (s1_id, cand_id), score in zip(val_pair_metadata, val_scores):
        s1_to_scored_cands[s1_id].append((cand_id, score))
        
    best_thresh = 0.5
    best_f05 = 0.0
    
    print("\nGrid searching classification threshold for macro F_0.5 optimization:", flush=True)
    for thresh in np.arange(0.30, 0.95, 0.05):
        pred_dict = {}
        for s1_id in val_s1_df['entity_id']:
            cands = s1_to_scored_cands.get(s1_id, [])
            matched = {cid for cid, s in cands if s >= thresh}
            pred_dict[s1_id] = matched
            
        macro_f05 = evaluate_macro_f05(val_gt, pred_dict)
        print(f"  Threshold {thresh:.2f} -> Macro F_0.5: {macro_f05:.4f}", flush=True)
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = thresh
            
    print(f"\n============================================================", flush=True)
    print(f"  >>> BEST VALIDATION MACRO F_0.5: {best_f05 * 100:.2f}% at Threshold = {best_thresh:.2f} <<<", flush=True)
    print(f"============================================================", flush=True)
    
    # Singletons accuracy
    final_pred_dict = {
        s1_id: {cid for cid, s in s1_to_scored_cands.get(s1_id, []) if s >= best_thresh}
        for s1_id in val_s1_df['entity_id']
    }
    singleton_s1s = [eid for eid, m in val_gt.items() if len(m) == 0]
    singleton_acc = np.mean([1.0 if len(final_pred_dict[eid]) == 0 else 0.0 for eid in singleton_s1s])
    print(f"Singleton Accuracy ({len(singleton_s1s)} entities): {singleton_acc * 100:.2f}%", flush=True)
    print(f"Total Experiment Time: {time.time()-t0:.2f} seconds", flush=True)

if __name__ == '__main__':
    run_experiment()
