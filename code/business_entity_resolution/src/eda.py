import os
import sys
import pandas as pd
import numpy as np

def run_eda():
    train_dir = '6ab10eb3b23ba_student_resource/student_resource/dataset/train'
    test_dir = '6ab10eb3b23ba_student_resource/student_resource/dataset/test'

    gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')
    s1_path = os.path.join(train_dir, 'train_source1.tsv')
    s2_path = os.path.join(train_dir, 'train_source2.tsv')
    s3_path = os.path.join(train_dir, 'train_source3.tsv')

    print("Loading ground truth...", flush=True)
    gt = pd.read_csv(gt_path, sep='\t')
    gt['match_count'] = gt['matched_entity_ids'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)

    print(f"Total S1 entities in Ground Truth: {len(gt):,}", flush=True)
    print("Match Count Distribution:\n", gt['match_count'].value_counts(normalize=True).head(10), flush=True)
    print("Match Count Absolute Counts:\n", gt['match_count'].value_counts().head(10), flush=True)
    print(f"Singleton fraction (0 matches): {(gt['match_count'] == 0).mean():.4f}", flush=True)

    print("\nReading head of sources to examine noisy examples...", flush=True)
    s1_df = pd.read_csv(s1_path, sep='\t', nrows=100000)
    s2_df = pd.read_csv(s2_path, sep='\t', nrows=100000)
    s3_df = pd.read_csv(s3_path, sep='\t', nrows=100000)

    s1_dict = s1_df.set_index('entity_id').to_dict('index')
    s2_dict = s2_df.set_index('entity_id').to_dict('index')
    s3_dict = s3_df.set_index('entity_id').to_dict('index')

    print("\n=== SAMPLE GROUND TRUTH MATCHES ===", flush=True)
    examples_shown = 0
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        if s1_id in s1_dict and row['match_count'] > 0:
            matches = row['matched_entity_ids'].split(',')
            found_matches = []
            for m in matches:
                if m.startswith('S2-') and m in s2_dict:
                    found_matches.append((m, s2_dict[m]))
                elif m.startswith('S3-') and m in s3_dict:
                    found_matches.append((m, s3_dict[m]))
            if found_matches:
                s1_info = s1_dict[s1_id]
                print(f"\n--- Entity S1 ({s1_id}) [Country: {s1_info['country']}] ---")
                print(f"  Name:    {s1_info['business_name']}")
                print(f"  Address: {s1_info['business_address']}")
                for mid, minfo in found_matches:
                    print(f"  --> MATCH {mid}:")
                    print(f"      Name:    {minfo['business_name']}")
                    print(f"      Address: {minfo['business_address']}")
                examples_shown += 1
                if examples_shown >= 10:
                    break

if __name__ == '__main__':
    run_eda()
