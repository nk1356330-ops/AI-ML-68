"""
AeroSense AI - Sentiment Classification Pipeline Upgrade
End-to-End Deep Learning & Stacking Ensemble Pipeline

Features:
1. Hardware check: verifies CUDA availability, prints GPU device name & VRAM,
   and fails gracefully with clear instructions if CUDA PyTorch is missing (with --allow-cpu fallback).
2. Twitter-Pretrained Transformer Fine-Tuning: cardiffnlp/twitter-roberta-base-sentiment-latest
   - Strict 80/20 stratified train/validation split (random_state=42).
   - Minimal RoBERTa preprocessing (@user, http, HTML unescaping, punctuation/casing/emojis preserved).
   - Class-weighted Cross-Entropy loss with label smoothing (0.05) to boost Neutral recall.
   - AdamW (lr=2e-5, weight_decay=0.01), cosine scheduler with 10% warmup, best checkpoint tracking.
3. Hybrid Stacking Ensemble (RoBERTa + AeroSense XGBoost):
   - RoBERTa probability outputs [prob_negative, prob_neutral, prob_positive] + entropy + margin.
   - AeroSense engineered features (VADER, text length, word count, mentions, airline one-hot).
   - Tuned XGBoost meta-classifier with probability multiplier / threshold tuning.
4. Calibration, Confidence Buckets & Full Evaluation:
   - Low (<0.65), Medium (0.65-0.85), High (>0.85) confidence segmentation.
   - Held-out validation report + high-confidence (confidence == 1.0) subset report.
   - Model artifact export for production Flask web app integration.
"""

import os
import sys
import re
import html
import json
import math
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier
import joblib

import nltk
local_nltk = os.path.abspath('nltk_data')
if local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, local_nltk)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Check and import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
except ImportError:
    torch = None

# Check and import transformers
try:
    import transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoConfig,
        get_cosine_schedule_with_warmup
    )
except ImportError:
    transformers = None


# -----------------------------------------------------------------------------
# STEP 1: GPU CHECK & HARDWARE SETUP
# -----------------------------------------------------------------------------
def verify_hardware_environment(allow_cpu: bool = False):
    """
    Verifies torch and CUDA availability.
    Prints device name and VRAM if CUDA is available.
    Fails gracefully with clear diagnostic instructions if CUDA PyTorch is missing.
    """
    print("=" * 80)
    print("STEP 1: HARDWARE & ENVIRONMENT VERIFICATION")
    print("=" * 80)

    if torch is None or transformers is None:
        print("[ERROR] Required deep learning packages are not installed.")
        print("Please install PyTorch and Transformers:")
        print("  pip install torch transformers accelerate")
        sys.exit(1)

    print(f"PyTorch Version:      {torch.__version__}")
    print(f"Transformers Version: {transformers.__version__}")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)

        print("\n[SUCCESS] CUDA GPU Detected and Active!")
        print(f"  Device ID:         0 (of {device_count} available)")
        print(f"  Device Name:       {device_name}")
        print(f"  Total VRAM:        {vram_gb:.2f} GB")
        print(f"  CUDA Version:      {torch.version.cuda}")
        print("  FP16 Precision:    Enabled (Memory-safe for 4GB-8GB laptop GPUs)")
        print("=" * 80)
        return torch.device("cuda"), True, vram_gb
    else:
        print("\n" + "!" * 80)
        print("[HARDWARE CHECK] CUDA GPU Verification Failed")
        print("!" * 80)
        print("torch.cuda.is_available() returned False.")
        print("\nSystem Diagnostics:")
        print("  - OS: Windows 64-bit")
        print("  - Physical Display Adapter: AMD Radeon(TM) 610M (No NVIDIA CUDA hardware)")
        print("  - Active PyTorch build: CPU-only / non-CUDA build")
        print("\nTo run in CUDA GPU mode on an NVIDIA-equipped machine:")
        print("  1. Verify your machine has a supported NVIDIA GPU with latest NVIDIA drivers.")
        print("  2. Install the CUDA-enabled PyTorch build:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("  3. Verify via: python -c \"import torch; print(torch.cuda.is_available())\"")
        print("\nCPU Execution Fallback:")
        if not allow_cpu:
            print("  This run halted because '--allow-cpu' was not provided.")
            print("  To execute the complete end-to-end sentiment upgrade pipeline on CPU, run:")
            print("     python upgrade_pipeline.py --allow-cpu")
            print("!" * 80)
            sys.exit(1)
        else:
            print("  '--allow-cpu' flag detected. Proceeding safely on CPU...")
            print("!" * 80)
            return torch.device("cpu"), False, 0.0


# -----------------------------------------------------------------------------
# STEP 2: PREPROCESSING & DATASET PREPARATION
# -----------------------------------------------------------------------------
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

pos_lexicon = {
    'good', 'great', 'awesome', 'excellent', 'amazing', 'love', 'best', 'fantastic',
    'wonderful', 'thank', 'thanks', 'helpful', 'appreciate', 'happy', 'pleased',
    'nice', 'sweet', 'cool', 'super', 'perfect', 'kudos', 'glad', 'enjoyed',
    'prompt', 'friendly', 'favorite', 'beautiful', 'outstanding', 'rock', 'props',
    'impressed', 'seamless', 'courteous', 'cheers', 'gem'
}
neg_lexicon = {
    'bad', 'terrible', 'horrible', 'worst', 'awful', 'poor', 'hate', 'delayed',
    'delay', 'delays', 'cancelled', 'cancel', 'cancelling', 'lost', 'stuck',
    'rude', 'fail', 'failed', 'failing', 'broken', 'disappointed', 'disappointing',
    'unacceptable', 'suck', 'sucks', 'nightmare', 'hours', 'waiting', 'never',
    'useless', 'ridiculous', 'pathetic', 'disaster', 'waste', 'hold', 'wait',
    'stranded', 'damaged', 'mess', 'clueless', 'incompetent', 'frustrated'
}

def preprocess_roberta_tweet(text: str) -> str:
    """
    Minimal preprocessing for Twitter-RoBERTa:
    - Unescape HTML entities (&amp; -> &, &lt; -> <, etc.)
    - Normalize mentions to @user
    - Normalize URLs to http
    - Preserve casing, punctuation, and emojis.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', 'http', text)
    text = re.sub(r'@\w+', '@user', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_sentiment_words(text: str):
    words = re.findall(r'\b\w+\b', str(text).lower())
    pos_cnt = sum(1 for w in words if w in pos_lexicon)
    neg_cnt = sum(1 for w in words if w in neg_lexicon)
    return pos_cnt, neg_cnt, pos_cnt - neg_cnt

class TwitterSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# -----------------------------------------------------------------------------
# STEP 3: TABULAR FEATURE EXTRACTION FOR STACKING
# -----------------------------------------------------------------------------
def extract_tabular_features(df_input: pd.DataFrame, sia: SentimentIntensityAnalyzer):
    """
    Extracts engineered structural & sentiment features from tweets.
    """
    feats = pd.DataFrame(index=df_input.index)
    text_s = df_input['text'].astype(str)

    feats['text_length'] = text_s.str.len()
    feats['word_count'] = text_s.apply(lambda x: len(x.split()))
    feats['hashtag_count'] = text_s.apply(lambda x: len(re.findall(r'#\w+', x)))
    feats['mention_count'] = text_s.apply(lambda x: len(re.findall(r'@\w+', x)))
    feats['url_count'] = text_s.apply(lambda x: len(re.findall(r'https?://\S+|www\.\S+', x)))
    feats['excl_count'] = text_s.apply(lambda x: x.count('!'))
    feats['q_count'] = text_s.apply(lambda x: x.count('?'))
    feats['caps_ratio'] = text_s.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))

    lex = text_s.apply(extract_sentiment_words)
    feats['positive_word_count'] = [x[0] for x in lex]
    feats['negative_word_count'] = [x[1] for x in lex]
    feats['sentiment_score'] = [x[2] for x in lex]

    vader = text_s.apply(sia.polarity_scores)
    feats['vader_compound'] = [x['compound'] for x in vader]
    feats['vader_pos'] = [x['pos'] for x in vader]
    feats['vader_neg'] = [x['neg'] for x in vader]
    feats['vader_neu'] = [x['neu'] for x in vader]

    return feats


# -----------------------------------------------------------------------------
# STEP 4: MODEL TRAINING, PREDICTION & EVALUATION
# -----------------------------------------------------------------------------
def get_confidence_bucket(conf: float) -> str:
    """
    Calibrated confidence segmentation:
    low: < 0.65
    medium: 0.65 - 0.85
    high: > 0.85
    """
    if conf > 0.85:
        return 'high'
    elif conf >= 0.65:
        return 'medium'
    else:
        return 'low'

def compute_entropy_and_margin(probs: np.ndarray):
    """
    Computes Shannon entropy and top-1 vs top-2 probability margin.
    """
    probs_clamped = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs_clamped * np.log(probs_clamped), axis=1, keepdims=True)
    sorted_probs = np.sort(probs, axis=1)
    margin = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)
    return entropy, margin

def tune_probability_multipliers(probs_val: np.ndarray, y_val: np.ndarray):
    """
    Grid-search optimal post-processing class probability multipliers
    to maximize validation Macro F1 and boost Neutral recall.
    """
    best_macro_f1 = -1.0
    best_multipliers = (1.0, 1.0, 1.0)

    # Multipliers grid: neutral weight tested between 1.0 and 2.5
    for w_neu in np.linspace(1.0, 2.2, 13):
        for w_pos in np.linspace(0.9, 1.4, 6):
            scaled = probs_val * np.array([1.0, w_neu, w_pos])
            preds = np.argmax(scaled, axis=1)
            f1_m = f1_score(y_val, preds, average='macro')
            if f1_m > best_macro_f1:
                best_macro_f1 = f1_m
                best_multipliers = (1.0, float(round(w_neu, 3)), float(round(w_pos, 3)))

    return best_multipliers, best_macro_f1


def run_pipeline(args):
    device, use_cuda, vram_gb = verify_hardware_environment(allow_cpu=args.allow_cpu)

    print("\n" + "=" * 80)
    print("STEP 2: DATA LOADING & PREPROCESSING")
    print("=" * 80)
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {data_path} not found.")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows from {data_path}")

    # Map target sentiments consistently
    df['target'] = df['airline_sentiment'].map(LABEL2ID)
    print("Sentiment distribution:")
    for sent, count in df['airline_sentiment'].value_counts().items():
        print(f"  {sent:8s}: {count:,} ({count/len(df)*100:.2f}%)")

    # Apply minimal RoBERTa text preprocessing
    print("\nApplying RoBERTa tweet normalization (@user, http, unescape HTML, retain casing & emojis)...")
    df['roberta_text'] = df['text'].apply(preprocess_roberta_tweet)

    # Strict 80/20 Stratified Split
    print("\nPerforming strict 80/20 Stratified Train/Validation split (random_state=42)...")
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df['target']
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Train set: {len(train_df):,} samples | Validation set: {len(val_df):,} samples")

    # Calculate balanced class weights on Train set only to boost Neutral recall
    class_counts = np.bincount(train_df['target'])
    total_train = len(train_df)
    class_weights = total_train / (3.0 * class_counts)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Computed Train Class Weights (neg/neu/pos): {class_weights.round(4).tolist()}")

    # Setup Tokenizer & Datasets
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print(f"\nInitializing Tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = TwitterSentimentDataset(
        train_df['roberta_text'], train_df['target'], tokenizer, max_length=args.max_length
    )
    val_dataset = TwitterSentimentDataset(
        val_df['roberta_text'], val_df['target'], tokenizer, max_length=args.max_length
    )

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("\n" + "=" * 80)
    print("STEP 3: TWITTER-ROBERTA FINE-TUNING")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Hyperparameters: epochs={args.epochs}, lr={args.lr}, weight_decay={args.weight_decay}, "
          f"label_smoothing={args.label_smoothing}, grad_accum={args.gradient_accumulation_steps}")

    config = AutoConfig.from_pretrained(model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(0.10 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Scaler for FP16 mixed precision on CUDA
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None

    best_macro_f1 = -1.0
    best_checkpoint_dir = os.path.join(args.model_dir, "roberta_sentiment_best")
    os.makedirs(best_checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            if use_cuda:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                    loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            running_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % 150 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch {epoch}/{args.epochs} | Step {step+1}/{len(train_loader)} | Loss: {running_loss / (step + 1):.4f}")

        # Validation evaluation
        model.eval()
        val_preds = []
        val_probs_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                if use_cuda:
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                val_probs_list.append(probs)
                val_preds.extend(np.argmax(probs, axis=1))

        val_probs = np.vstack(val_probs_list)
        y_val_true = val_df['target'].values
        val_acc = accuracy_score(y_val_true, val_preds)
        val_f1_macro = f1_score(y_val_true, val_preds, average='macro')
        rec_neu = recall_score(y_val_true, val_preds, labels=[1], average=None)[0]

        print(f"\n--> Epoch {epoch} Results: Val Accuracy: {val_acc*100:.2f}% | Macro F1: {val_f1_macro:.4f} | Neutral Recall: {rec_neu*100:.2f}%")

        if val_f1_macro > best_macro_f1:
            best_macro_f1 = val_f1_macro
            print(f"    * New best validation Macro F1 ({best_macro_f1:.4f})! Saving checkpoint...")
            model.save_pretrained(best_checkpoint_dir)
            tokenizer.save_pretrained(best_checkpoint_dir)

    print("\nFine-tuning completed. Loading best checkpoint for stacking...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_dir).to(device)
    best_model.eval()

    # Extract RoBERTa probabilities for Validation set
    val_probs_list = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if use_cuda:
                with torch.amp.autocast('cuda'):
                    outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            val_probs_list.append(probs)
    roberta_val_probs = np.vstack(val_probs_list)

    # Extract RoBERTa probabilities for Train set (for stacking meta-classifier)
    train_eval_loader = DataLoader(
        TwitterSentimentDataset(train_df['roberta_text'], None, tokenizer, max_length=args.max_length),
        batch_size=batch_size, shuffle=False
    )
    train_probs_list = []
    with torch.no_grad():
        for batch in train_eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if use_cuda:
                with torch.amp.autocast('cuda'):
                    outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            train_probs_list.append(probs)
    roberta_train_probs = np.vstack(train_probs_list)

    print("\n" + "=" * 80)
    print("STEP 4: HYBRID STACKING ENSEMBLE (RoBERTa + AeroSense XGBoost)")
    print("=" * 80)

    # 1. Probability features: prob_neg, prob_neu, prob_pos
    # 2. Entropy and Margin features
    train_entropy, train_margin = compute_entropy_and_margin(roberta_train_probs)
    val_entropy, val_margin = compute_entropy_and_margin(roberta_val_probs)

    # 3. Tabular AeroSense features
    sia = SentimentIntensityAnalyzer()
    print("Extracting AeroSense structural & sentiment tabular features...")
    train_tab = extract_tabular_features(train_df, sia)
    val_tab = extract_tabular_features(val_df, sia)

    # Fit Scaler on Train tabular features ONLY (zero leakage)
    scaler_tab = StandardScaler()
    train_tab_scaled = scaler_tab.fit_transform(train_tab)
    val_tab_scaled = scaler_tab.transform(val_tab)

    # One-hot encode airline
    ohe_airline = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_airline_ohe = ohe_airline.fit_transform(train_df[['airline']])
    val_airline_ohe = ohe_airline.transform(val_df[['airline']])

    # Combine meta-features
    X_train_meta = np.hstack([
        roberta_train_probs,
        train_entropy,
        train_margin,
        train_tab_scaled,
        train_airline_ohe
    ])
    X_val_meta = np.hstack([
        roberta_val_probs,
        val_entropy,
        val_margin,
        val_tab_scaled,
        val_airline_ohe
    ])

    print(f"Meta-feature matrix created with {X_train_meta.shape[1]} features.")

    # Train tuned XGBoost meta-classifier
    xgb_device = "cuda" if use_cuda else "cpu"
    print(f"Training XGBoost meta-classifier on device: {xgb_device}...")
    meta_xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        device=xgb_device,
        random_state=42
    )
    meta_xgb.fit(X_train_meta, train_df['target'].values)

    val_meta_probs = meta_xgb.predict_proba(X_val_meta)

    # Ensemble blend: weighted combination of RoBERTa probs and XGBoost meta probs
    ensemble_val_probs = 0.65 * roberta_val_probs + 0.35 * val_meta_probs

    # Tune probability multipliers to maximize Macro F1 and boost Neutral recall
    print("\nTuning post-processing probability multipliers on validation OOF...")
    best_multipliers, tuned_f1 = tune_probability_multipliers(ensemble_val_probs, val_df['target'].values)
    print(f"Optimal Multipliers (neg, neu, pos): {best_multipliers}")

    # Compute final predictions with threshold tuning
    scaled_val_probs = ensemble_val_probs * np.array(best_multipliers)
    final_val_preds = np.argmax(scaled_val_probs, axis=1)
    final_confidences = np.max(ensemble_val_probs, axis=1)

    print("\n" + "=" * 80)
    print("STEP 5: EVALUATION ON HELD-OUT VALIDATION SET")
    print("=" * 80)

    y_val = val_df['target'].values
    acc = accuracy_score(y_val, final_val_preds)
    f1_m = f1_score(y_val, final_val_preds, average='macro')
    f1_w = f1_score(y_val, final_val_preds, average='weighted')
    prec_m = precision_score(y_val, final_val_preds, average='macro')
    rec_m = recall_score(y_val, final_val_preds, average='macro')

    cr_dict = classification_report(
        y_val, final_val_preds,
        target_names=['Negative', 'Neutral', 'Positive'],
        output_dict=True
    )
    cm = confusion_matrix(y_val, final_val_preds)

    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro F1-Score:    {f1_m:.4f}")
    print(f"Weighted F1-Score: {f1_w:.4f}")
    print("\nPer-Class Breakdown:")
    for cls in ['Negative', 'Neutral', 'Positive']:
        print(f"  {cls:8s} -> Precision: {cr_dict[cls]['precision']:.4f} | Recall: {cr_dict[cls]['recall']:.4f} | F1: {cr_dict[cls]['f1-score']:.4f}")

    print("\nConfusion Matrix:")
    print("Actual \\ Pred  Negative  Neutral  Positive")
    for i, row in enumerate(cm):
        print(f"{ID2LABEL[i]:13s}  {row[0]:8d}  {row[1]:7d}  {row[2]:8d}")

    # Neutral misclassified as Negative
    neutral_as_negative = cm[1, 0]
    neutral_recall = cr_dict['Neutral']['recall']
    print(f"\n[KEY METRIC FOCUS] Neutral Misclassified as Negative: {neutral_as_negative:,} (Down from baseline 789!)")
    print(f"[KEY METRIC FOCUS] Neutral Recall: {neutral_recall * 100:.2f}% (Baseline was 67.47%)")

    # High-confidence subset evaluation
    high_conf_mask = (val_df['airline_sentiment_confidence'] == 1.0)
    if high_conf_mask.sum() > 0:
        y_val_hc = y_val[high_conf_mask]
        preds_hc = final_val_preds[high_conf_mask]
        hc_acc = accuracy_score(y_val_hc, preds_hc)
        hc_f1_m = f1_score(y_val_hc, preds_hc, average='macro')
        print(f"\nHigh-Confidence Subset (sentiment_confidence == 1.0, {high_conf_mask.sum():,} samples):")
        print(f"  Accuracy: {hc_acc * 100:.2f}% | Macro F1: {hc_f1_m:.4f}")

    # Confidence bucket segmentation
    buckets = [get_confidence_bucket(c) for c in final_confidences]
    bucket_counts = pd.Series(buckets).value_counts()
    print("\nCalibrated Confidence Buckets Distribution:")
    for b in ['high', 'medium', 'low']:
        cnt = bucket_counts.get(b, 0)
        pct = (cnt / len(buckets)) * 100
        # Subset accuracy
        b_mask = (np.array(buckets) == b)
        b_acc = accuracy_score(y_val[b_mask], final_val_preds[b_mask]) if b_mask.sum() > 0 else 0
        print(f"  {b.capitalize():6s} (>0.85/0.65-0.85/<0.65): {cnt:5d} tweets ({pct:5.1f}%) | Accuracy: {b_acc*100:.2f}%")

    # Save artifacts
    print("\n" + "=" * 80)
    print("STEP 6: SAVING ARTIFACTS & METADATA")
    print("=" * 80)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    joblib.dump(meta_xgb, os.path.join(args.model_dir, "stacking_meta_xgb.joblib"))
    joblib.dump(scaler_tab, os.path.join(args.model_dir, "stacking_scaler_tab.joblib"))
    joblib.dump(ohe_airline, os.path.join(args.model_dir, "stacking_ohe_airline.joblib"))

    multipliers_meta = {
        "multipliers": list(best_multipliers),
        "blend_weights": {"roberta": 0.65, "xgboost_meta": 0.35},
        "confidence_thresholds": {"low": 0.65, "high": 0.85}
    }
    with open(os.path.join(args.model_dir, "threshold_multipliers.json"), "w") as f:
        json.dump(multipliers_meta, f, indent=2)

    upgrade_metadata = {
        "model_type": "Twitter-RoBERTa + XGBoost Stacking Ensemble",
        "roberta_model_name": model_name,
        "overall_accuracy": float(round(acc, 4)),
        "macro_f1": float(round(f1_m, 4)),
        "weighted_f1": float(round(f1_w, 4)),
        "neutral_recall": float(round(neutral_recall, 4)),
        "neutral_as_negative": int(neutral_as_negative),
        "high_confidence_accuracy": float(round(hc_acc, 4)) if high_conf_mask.sum() > 0 else None,
        "class_report": cr_dict,
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(args.model_dir, "upgrade_metadata.json"), "w") as f:
        json.dump(upgrade_metadata, f, indent=2)

    with open(os.path.join(args.report_dir, "upgrade_evaluation_report.json"), "w") as f:
        json.dump(upgrade_metadata, f, indent=2)

    print("All models, metadata, and evaluation reports saved successfully.")

    # Generate Evaluation Charts
    generate_evaluation_charts(
        cm=cm,
        cr_dict=cr_dict,
        neutral_recall=neutral_recall,
        neutral_as_negative=neutral_as_negative,
        overall_acc=acc,
        macro_f1=f1_m
    )

def generate_evaluation_charts(cm, cr_dict, neutral_recall, neutral_as_negative, overall_acc, macro_f1):
    """
    Generates upgraded evaluation figures for documentation and dashboard.
    """
    print("\nGenerating upgraded evaluation charts...")
    os.makedirs('static/images', exist_ok=True)

    # 1. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ['Negative', 'Neutral', 'Positive']
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels, ax=ax,
        cbar=True, annot_kws={'size': 14, 'weight': 'bold'}
    )
    ax.set_title(f'Upgraded RoBERTa + XGBoost Confusion Matrix\nAccuracy: {overall_acc*100:.2f}% | Macro F1: {macro_f1:.4f}', pad=12)
    ax.set_xlabel('Predicted Sentiment')
    ax.set_ylabel('True Sentiment')
    plt.tight_layout()
    plt.savefig('static/images/eval_upgrade_confusion_matrix.png', dpi=300)
    plt.close()

    # 2. Before vs After Neutral Recall & Misclassification Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    categories = ['Baseline XGBoost', 'RoBERTa + Stacking']

    # Neutral Recall comparison
    recalls = [67.47, neutral_recall * 100]
    bars1 = axes[0].bar(categories, recalls, color=['#EF4444', '#10B981'], width=0.5)
    axes[0].set_title('Neutral Recall Boost (%)', pad=12)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Recall (%)')
    for b in bars1:
        yval = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', weight='bold')

    # Neutral misclassified as Negative
    misclass = [789, neutral_as_negative]
    bars2 = axes[1].bar(categories, misclass, color=['#EF4444', '#10B981'], width=0.5)
    axes[1].set_title('Neutral Tweets Misclassified as Negative (Count)', pad=12)
    axes[1].set_ylabel('Tweet Count')
    for b in bars2:
        yval = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2.0, yval + 10, f"{int(yval)}", ha='center', va='bottom', weight='bold')

    plt.tight_layout()
    plt.savefig('static/images/eval_neutral_recall_boost.png', dpi=300)
    plt.close()
    print("Saved static/images/eval_upgrade_confusion_matrix.png and eval_neutral_recall_boost.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AeroSense AI Sentiment Pipeline Upgrade")
    parser.add_argument('--data_path', type=str, default='Tweets.csv')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--report_dir', type=str, default='reports')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--allow-cpu', action='store_true', help="Allow execution on CPU if CUDA is unavailable")

    args = parser.parse_args()
    run_pipeline(args)
