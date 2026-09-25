import numpy as np

def compute_f_beta(precision, recall, beta=0.5):
    """
    Computes F_beta score for a single entity.
    beta=0.5: (1 + 0.25) * P * R / (0.25 * P + R) = 1.25 * P * R / (0.25 * P + R)
    """
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denom

def evaluate_macro_f05(ground_truth_dict, predictions_dict):
    """
    Computes macro-averaged F_0.5 score across all Source 1 entities in ground_truth_dict.
    ground_truth_dict: {s1_id: set([matched_s2_or_s3_ids])}
    predictions_dict:  {s1_id: set([predicted_s2_or_s3_ids])}
    
    Singletons:
      - If ground truth is empty set and prediction is empty set -> F_0.5 = 1.0
      - If ground truth is empty set and prediction is non-empty -> F_0.5 = 0.0
    """
    scores = []
    
    for s1_id, true_matches in ground_truth_dict.items():
        pred_matches = predictions_dict.get(s1_id, set())
        
        n_true = len(true_matches)
        n_pred = len(pred_matches)
        
        if n_true == 0:
            if n_pred == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)
            continue
            
        if n_pred == 0:
            scores.append(0.0)
            continue
            
        tp = len(true_matches & pred_matches)
        precision = tp / n_pred
        recall = tp / n_true
        
        f05 = compute_f_beta(precision, recall, beta=0.5)
        scores.append(f05)
        
    macro_f05 = float(np.mean(scores))
    return macro_f05

def compute_blocking_recall(ground_truth_dict, candidates_dict):
    """
    Computes candidate generation (blocking) recall:
    The fraction of true matches present in the candidate sets.
    """
    total_true = 0
    total_captured = 0
    s1_full_recalls = []
    
    for s1_id, true_matches in ground_truth_dict.items():
        if not true_matches:
            continue
        cands = set(candidates_dict.get(s1_id, []))
        captured = len(true_matches & cands)
        total_true += len(true_matches)
        total_captured += captured
        s1_full_recalls.append(captured / len(true_matches))
        
    overall_recall = total_captured / total_true if total_true > 0 else 1.0
    macro_recall = float(np.mean(s1_full_recalls)) if s1_full_recalls else 1.0
    return overall_recall, macro_recall
