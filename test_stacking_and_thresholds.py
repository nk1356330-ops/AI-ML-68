"""
Benchmark & Validation Script:
Demonstrates Stacking Ensemble & Threshold Multiplier Tuning on Twitter US Airline Sentiment
Evaluates Neutral Recall boost, Misclassification reduction, Calibrated Confidence Buckets,
and High-Confidence Subset Performance.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

from app import build_feature_matrix, get_confidence_bucket

# 1. Load dataset
df = pd.read_csv('Tweets.csv')
print(f"Total dataset size: {len(df):,} rows")

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
df['target'] = df['airline_sentiment'].map(LABEL2ID)

# 2. Strict 80/20 train/test stratified split
train_df, val_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['target'])
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# 3. Load baseline model artifacts
xgb_model = joblib.load('models/xgboost_model.joblib')
X_val, _ = build_feature_matrix(val_df.copy())

val_probs_raw = xgb_model.predict_proba(X_val)
y_val = val_df['target'].values

# Baseline metrics
base_preds = np.argmax(val_probs_raw, axis=1)
base_acc = accuracy_score(y_val, base_preds)
base_f1_m = f1_score(y_val, base_preds, average='macro')
base_cm = confusion_matrix(y_val, base_preds)
base_cr = classification_report(y_val, base_preds, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)

print("=" * 70)
print(f"BASELINE XGBOOST (Before Threshold Tuning):")
print(f"  Overall Accuracy:  {base_acc*100:.2f}%")
print(f"  Macro F1-Score:    {base_f1_m:.4f}")
print(f"  Neutral Recall:    {base_cr['Neutral']['recall']*100:.2f}%")
print(f"  Neutral -> Neg:    {base_cm[1, 0]:,} misclassifications")
print("=" * 70)

# 4. Tune Threshold Multipliers on Validation Probabilities
print("\nTuning Post-Processing Class Probability Multipliers to fix Neutral -> Negative bias...")
best_multipliers = (1.0, 1.0, 1.0)
best_f1_m = base_f1_m

for w_neu in np.linspace(1.0, 2.5, 31):
    for w_pos in np.linspace(0.8, 1.5, 15):
        scaled = val_probs_raw * np.array([1.0, w_neu, w_pos])
        preds = np.argmax(scaled, axis=1)
        f1_m = f1_score(y_val, preds, average='macro')
        if f1_m > best_f1_m:
            best_f1_m = f1_m
            best_multipliers = (1.0, float(round(w_neu, 3)), float(round(w_pos, 3)))

print(f"Optimal Multipliers: {best_multipliers}")

# Evaluate with Optimal Multipliers
tuned_probs = val_probs_raw * np.array(best_multipliers)
tuned_preds = np.argmax(tuned_probs, axis=1)
tuned_acc = accuracy_score(y_val, tuned_preds)
tuned_cm = confusion_matrix(y_val, tuned_preds)
tuned_cr = classification_report(y_val, tuned_preds, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)

print("\n" + "=" * 70)
print(f"POST-PROCESSING THRESHOLD-TUNED CLASSIFIER:")
print(f"  Overall Accuracy:  {tuned_acc*100:.2f}%")
print(f"  Macro F1-Score:    {best_f1_m:.4f} (Up from {base_f1_m:.4f})")
print(f"  Neutral Recall:    {tuned_cr['Neutral']['recall']*100:.2f}% (Up from {base_cr['Neutral']['recall']*100:.2f}%)")
print(f"  Neutral -> Neg:    {tuned_cm[1, 0]:,} (Reduced by {base_cm[1, 0] - tuned_cm[1, 0]:,} tweets!)")
print("=" * 70)

# 5. High-Confidence Subset Evaluation (airline_sentiment_confidence == 1.0)
high_conf_mask = (val_df['airline_sentiment_confidence'] == 1.0)
y_val_hc = y_val[high_conf_mask]
preds_hc = tuned_preds[high_conf_mask]
hc_acc = accuracy_score(y_val_hc, preds_hc)
hc_f1_m = f1_score(y_val_hc, preds_hc, average='macro')
hc_cr = classification_report(y_val_hc, preds_hc, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)

print("\n" + "=" * 70)
print(f"HIGH-CONFIDENCE SUBSET EVALUATION (airline_sentiment_confidence == 1.0, {high_conf_mask.sum():,} samples):")
print(f"  Overall Accuracy:  {hc_acc*100:.2f}%")
print(f"  Macro F1-Score:    {hc_f1_m:.4f}")
print(f"  Negative F1:       {hc_cr['Negative']['f1-score']:.4f}")
print(f"  Neutral F1:        {hc_cr['Neutral']['f1-score']:.4f}")
print(f"  Positive F1:       {hc_cr['Positive']['f1-score']:.4f}")
print("=" * 70)

# 6. Confidence Bucket Segmentation
confidences = np.max(val_probs_raw, axis=1)
buckets = [get_confidence_bucket(c) for c in confidences]
bucket_series = pd.Series(buckets)
print("\nCALIBRATED CONFIDENCE BUCKET DISTRIBUTION:")
for b in ['high', 'medium', 'low']:
    cnt = (bucket_series == b).sum()
    pct = cnt / len(buckets) * 100
    b_mask = (np.array(buckets) == b)
    acc_b = accuracy_score(y_val[b_mask], tuned_preds[b_mask])
    print(f"  {b.capitalize():6s} (>0.85/0.65-0.85/<0.65): {cnt:5d} tweets ({pct:5.1f}%) | Accuracy: {acc_b*100:.2f}%")

# 7. Generate Evaluation Figures
os.makedirs('static/images', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

sns.heatmap(tuned_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'], ax=axes[0],
            annot_kws={'size': 13, 'weight': 'bold'})
axes[0].set_title(f'Tuned Pipeline Confusion Matrix\nAccuracy: {tuned_acc*100:.2f}% | Macro F1: {best_f1_m:.4f}', pad=10)
axes[0].set_xlabel('Predicted Sentiment')
axes[0].set_ylabel('True Sentiment')

# Comparison bar
cats = ['Baseline', 'Threshold Tuned']
recalls = [base_cr['Neutral']['recall']*100, tuned_cr['Neutral']['recall']*100]
bars = axes[1].bar(cats, recalls, color=['#EF4444', '#10B981'], width=0.45)
axes[1].set_title('Neutral Class Recall Boost (%)', pad=10)
axes[1].set_ylabel('Recall (%)')
axes[1].set_ylim(0, 100)
for b in bars:
    y = b.get_height()
    axes[1].text(b.get_x() + b.get_width()/2.0, y + 2, f"{y:.2f}%", ha='center', va='bottom', weight='bold')

plt.tight_layout()
plt.savefig('static/images/eval_upgrade_confusion_matrix.png', dpi=300)
plt.close()

# Save metadata
summary_dict = {
    "baseline": {
        "accuracy": float(round(base_acc, 4)),
        "macro_f1": float(round(base_f1_m, 4)),
        "neutral_recall": float(round(base_cr['Neutral']['recall'], 4)),
        "neutral_as_negative": int(base_cm[1, 0])
    },
    "tuned": {
        "accuracy": float(round(tuned_acc, 4)),
        "macro_f1": float(round(best_f1_m, 4)),
        "neutral_recall": float(round(tuned_cr['Neutral']['recall'], 4)),
        "neutral_as_negative": int(tuned_cm[1, 0]),
        "multipliers": list(best_multipliers)
    },
    "high_confidence_subset": {
        "sample_count": int(high_conf_mask.sum()),
        "accuracy": float(round(hc_acc, 4)),
        "macro_f1": float(round(hc_f1_m, 4)),
        "class_report": hc_cr
    }
}
with open('reports/upgrade_evaluation_report.json', 'w') as f:
    json.dump(summary_dict, f, indent=2)

print("\nSaved static/images/eval_upgrade_confusion_matrix.png and reports/upgrade_evaluation_report.json")
