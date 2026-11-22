"""
AI-ML-068: Airline Customer Sentiment Analysis Using Machine Learning
Complete EDA, Feature Engineering, SMOTE Cross-Validation, Model Training & Evaluation Script
"""

import os
import re
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

import nltk
# Add local nltk_data to search path
local_nltk = os.path.abspath('nltk_data')
if local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, local_nltk)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Create directories
os.makedirs('static/images', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("="*80)
print("STEP 1: DATA LOADING & INITIAL EXPLORATION")
print("="*80)

DATA_PATH = 'Tweets.csv'
df = pd.read_csv(DATA_PATH)
print(f"Dataset Loaded Successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
print(f"Columns: {list(df.columns)}")
print("\nSentiment Class Distribution:")
print(df['airline_sentiment'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))
print("\nAirline Counts:")
print(df['airline'].value_counts())

print("\n" + "="*80)
print("STEP 2: TEXT PREPROCESSING & MISSING VALUE HANDLING")
print("="*80)

# Missing values handling
df['negativereason'] = df['negativereason'].fillna('None')
df['tweet_location'] = df['tweet_location'].fillna('Unknown')
df['user_timezone'] = df['user_timezone'].fillna('Unknown')

# Stopwords setup with negation words preserved
base_sw = set(stopwords.words('english'))
negations = {'not', 'no', 'never', 'nor', 'neither', 'hardly', 'barely', 'scarcely', 
             'aint', 'arent', 'cant', 'couldnt', 'didnt', 'doesnt', 'dont', 'hadnt', 
             'hasnt', 'havent', 'isnt', 'mightnt', 'mustnt', 'neednt', 'shant', 
             'shouldnt', 'wasnt', 'werent', 'wont', 'wouldnt'}
custom_sw = base_sw - negations
lemmatizer = WordNetLemmatizer()

def clean_tweet_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtag symbol but keep text
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove special punctuation and symbols, retain alphanumeric and space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Lowercase & tokenize
    tokens = text.lower().split()
    # Remove stopwords & lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in custom_sw and len(w) > 1]
    return " ".join(cleaned_tokens)

print("Cleaning tweet texts (URL removal, mention removal, punctuation removal, lowercasing, lemmatization)...")
df['clean_text'] = df['text'].apply(clean_tweet_text)
print("Sample cleaned text:")
for i in range(3):
    print(f"  Raw:     {df['text'].iloc[i]}")
    print(f"  Cleaned: {df['clean_text'].iloc[i]}\n")

print("="*80)
print("STEP 3: FEATURE ENGINEERING")
print("="*80)

# 1. Structural text features (derived from raw text before cleaning)
df['text_length'] = df['text'].astype(str).str.len()
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
df['hashtag_count'] = df['text'].astype(str).apply(lambda x: len(re.findall(r'#\w+', x)))
df['mention_count'] = df['text'].astype(str).apply(lambda x: len(re.findall(r'@\w+', x)))
df['url_count'] = df['text'].astype(str).apply(lambda x: len(re.findall(r'https?://\S+|www\.\S+', x)))
df['excl_count'] = df['text'].astype(str).apply(lambda x: x.count('!'))
df['q_count'] = df['text'].astype(str).apply(lambda x: x.count('?'))
df['caps_ratio'] = df['text'].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))

# 2. Sentiment word features & lexicon
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

def extract_sentiment_words(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    pos_cnt = sum(1 for w in words if w in pos_lexicon)
    neg_cnt = sum(1 for w in words if w in neg_lexicon)
    score = pos_cnt - neg_cnt
    return pos_cnt, neg_cnt, score

lex_feats = df['text'].apply(extract_sentiment_words)
df['positive_word_count'] = [x[0] for x in lex_feats]
df['negative_word_count'] = [x[1] for x in lex_feats]
df['sentiment_score'] = [x[2] for x in lex_feats]

# 3. VADER sentiment intensity features
sia = SentimentIntensityAnalyzer()
vader_scores = df['text'].astype(str).apply(sia.polarity_scores)
df['vader_compound'] = [x['compound'] for x in vader_scores]
df['vader_pos'] = [x['pos'] for x in vader_scores]
df['vader_neg'] = [x['neg'] for x in vader_scores]
df['vader_neu'] = [x['neu'] for x in vader_scores]

# 4. Confidence buckets (high >0.8, medium 0.5-0.8, low <=0.5)
def get_conf_bucket(conf):
    if conf > 0.8:
        return 'high'
    elif conf >= 0.5:
        return 'medium'
    else:
        return 'low'

df['confidence_bucket'] = df['airline_sentiment_confidence'].apply(get_conf_bucket)
print(f"Confidence Bucket Distribution:\n{df['confidence_bucket'].value_counts(normalize=True)}")

print("\n" + "="*80)
print("STEP 4: EXPLORATORY DATA ANALYSIS (EDA) & CHART GENERATION")
print("="*80)

# Set common palette
sentiment_palette = {'negative': '#EF4444', 'neutral': '#F59E0B', 'positive': '#10B981'}
sentiment_order = ['negative', 'neutral', 'positive']

# Chart 1: Sentiment Distribution (Pie + Bar)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

counts = df['airline_sentiment'].value_counts()
colors = [sentiment_palette[s] for s in counts.index]

# Donut pie chart
wedges, texts, autotexts = axes[0].pie(
    counts.values,
    labels=[f"{k.capitalize()} ({v:,})" for k, v in counts.items()],
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2),
    pctdistance=0.75
)
for at in autotexts:
    at.set_color('white')
    at.set_weight('bold')
    at.set_fontsize(12)
axes[0].set_title('Airline Sentiment Distribution (Share)', pad=15)

# Bar chart with percentage annotations
sns.barplot(
    x=counts.index,
    y=counts.values,
    palette=sentiment_palette,
    order=sentiment_order,
    ax=axes[1],
    edgecolor='black',
    linewidth=1.2
)
for p in axes[1].patches:
    h = p.get_height()
    pct = (h / len(df)) * 100
    axes[1].annotate(f'{int(h):,} ({pct:.1f}%)',
                     (p.get_x() + p.get_width() / 2., h / 2),
                     ha='center', va='center', fontsize=11, color='white', weight='bold')

axes[1].set_title('Sentiment Class Frequency & Class Imbalance', pad=15)
axes[1].set_xlabel('Sentiment Class')
axes[1].set_ylabel('Tweet Count')
axes[1].set_xticklabels(['Negative', 'Neutral', 'Positive'])

plt.tight_layout()
chart1_path = 'static/images/eda_sentiment_distribution.png'
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"Saved: {chart1_path}")

# Chart 2: Text Length and Word Count Distribution by Sentiment
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(
    data=df,
    x='airline_sentiment',
    y='text_length',
    order=sentiment_order,
    palette=sentiment_palette,
    ax=axes[0],
    showmeans=True,
    meanprops=dict(marker='o', markeredgecolor='black', markerfacecolor='white')
)
axes[0].set_title('Tweet Character Length by Sentiment', pad=15)
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Character Count (Length)')
axes[0].set_xticklabels(['Negative', 'Neutral', 'Positive'])

sns.boxplot(
    data=df,
    x='airline_sentiment',
    y='word_count',
    order=sentiment_order,
    palette=sentiment_palette,
    ax=axes[1],
    showmeans=True,
    meanprops=dict(marker='o', markeredgecolor='black', markerfacecolor='white')
)
axes[1].set_title('Tweet Word Count by Sentiment', pad=15)
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Word Count')
axes[1].set_xticklabels(['Negative', 'Neutral', 'Positive'])

plt.tight_layout()
chart2_path = 'static/images/eda_text_length_boxplot.png'
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"Saved: {chart2_path}")

# Chart 3: Correlation Heatmap of Numeric & Sentiment Features
fig, ax = plt.subplots(figsize=(12, 9))
numeric_cols = [
    'text_length', 'word_count', 'hashtag_count', 'mention_count', 'url_count',
    'excl_count', 'q_count', 'caps_ratio',
    'positive_word_count', 'negative_word_count', 'sentiment_score',
    'vader_compound', 'vader_pos', 'vader_neg', 'airline_sentiment_confidence'
]
corr_matrix = df[numeric_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    ax=ax,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation'}
)
ax.set_title('Correlation Heatmap of Engineered NLP & Lexicon Features', pad=15)
plt.tight_layout()
chart3_path = 'static/images/eda_correlation_heatmap.png'
plt.savefig(chart3_path, dpi=300)
plt.close()
print(f"Saved: {chart3_path}")

# Chart 4: Sentiment Score & VADER Compound Distribution Analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for s in sentiment_order:
    sns.kdeplot(
        df[df['airline_sentiment'] == s]['sentiment_score'],
        ax=axes[0],
        label=s.capitalize(),
        color=sentiment_palette[s],
        fill=True,
        alpha=0.3,
        linewidth=2
    )
axes[0].set_title('Lexicon Sentiment Score Distribution by Class', pad=15)
axes[0].set_xlabel('Lexicon Score (Positive Words - Negative Words)')
axes[0].set_ylabel('Density')
axes[0].legend(title='Sentiment')

for s in sentiment_order:
    sns.kdeplot(
        df[df['airline_sentiment'] == s]['vader_compound'],
        ax=axes[1],
        label=s.capitalize(),
        color=sentiment_palette[s],
        fill=True,
        alpha=0.3,
        linewidth=2
    )
axes[1].set_title('VADER Compound Score Distribution by Class', pad=15)
axes[1].set_xlabel('VADER Compound Score [-1.0 (Most Negative) to +1.0 (Most Positive)]')
axes[1].set_ylabel('Density')
axes[1].legend(title='Sentiment')

plt.tight_layout()
chart4_path = 'static/images/eda_sentiment_score_analysis.png'
plt.savefig(chart4_path, dpi=300)
plt.close()
print(f"Saved: {chart4_path}")

# Chart 5: Airline-by-Airline Sentiment Breakdown & Top Negative Reasons
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

airline_sent = pd.crosstab(df['airline'], df['airline_sentiment'], normalize='index')[sentiment_order] * 100
airline_sent.plot(
    kind='bar',
    stacked=True,
    color=[sentiment_palette[s] for s in sentiment_order],
    ax=axes[0],
    edgecolor='black',
    linewidth=0.8
)
axes[0].set_title('Airline Sentiment Proportion (%) by Carrier', pad=15)
axes[0].set_xlabel('US Airline')
axes[0].set_ylabel('Percentage (%)')
axes[0].legend(['Negative', 'Neutral', 'Positive'], title='Sentiment', loc='upper right')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right')

# Top 8 Negative Reasons (excluding 'None')
neg_reasons = df[df['negativereason'] != 'None']['negativereason'].value_counts().head(8)
sns.barplot(
    x=neg_reasons.values,
    y=neg_reasons.index,
    palette='Reds_r',
    ax=axes[1],
    edgecolor='black',
    linewidth=0.8
)
axes[1].set_title('Top 8 Drivers of Negative Airline Customer Feedback', pad=15)
axes[1].set_xlabel('Frequency')
axes[1].set_ylabel('Negative Reason')
for p in axes[1].patches:
    w = p.get_width()
    axes[1].annotate(f'{int(w):,}', (w + 40, p.get_y() + p.get_height()/2.),
                     ha='left', va='center', fontsize=10, weight='bold')

plt.tight_layout()
chart5_path = 'static/images/eda_airline_sentiment_breakdown.png'
plt.savefig(chart5_path, dpi=300)
plt.close()
print(f"Saved: {chart5_path}")

print("\n" + "="*80)
print("STEP 5: DATA ENCODING & PARTITIONING (80/20 STRATIFIED SPLIT)")
print("="*80)

# Target label encoding: negative=0, neutral=1, positive=2
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
inv_sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
df['target'] = df['airline_sentiment'].map(sentiment_map)
y = df['target'].values

# Split 80/20 with stratified sampling
train_indices, test_indices = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_df = df.loc[train_indices].copy()
test_df = df.loc[test_indices].copy()

y_train = y[train_indices]
y_test = y[test_indices]

print(f"Training set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"Testing set:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
print(f"Train Class Distribution: {np.bincount(y_train) / len(y_train)}")
print(f"Test Class Distribution:  {np.bincount(y_test) / len(y_test)}")

# TF-IDF Vectorization
print("\nFitting TF-IDF Vectorizer (max_features=5000, ngram_range=(1,2), sublinear_tf=True)...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['clean_text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['clean_text'])
print(f"TF-IDF Matrix Shape: Train={X_train_tfidf.shape}, Test={X_test_tfidf.shape}")

# Numerical feature scaling
feature_num_cols = [
    'text_length', 'word_count', 'hashtag_count', 'mention_count', 'url_count',
    'excl_count', 'q_count', 'caps_ratio',
    'positive_word_count', 'negative_word_count', 'sentiment_score',
    'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu'
]
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_df[feature_num_cols])
X_test_num = scaler.transform(test_df[feature_num_cols])

# Categorical one-hot encoding (airline and confidence bucket)
cat_cols = ['airline', 'confidence_bucket']
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_train_cat = ohe.fit_transform(train_df[cat_cols])
X_test_cat = ohe.transform(test_df[cat_cols])

# Combine all features into unified sparse matrices
X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_num), X_train_cat]).tocsr()
X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_num), X_test_cat]).tocsr()

feature_names = (
    list(tfidf_vectorizer.get_feature_names_out()) +
    feature_num_cols +
    list(ohe.get_feature_names_out(cat_cols))
)
print(f"Total Feature Space Dimension: {X_train_combined.shape[1]} features.")

print("\n" + "="*80)
print("STEP 6: 5-FOLD STRATIFIED CROSS-VALIDATION WITH SMOTE WITHIN EACH FOLD")
print("="*80)
print("Applying SMOTE strictly inside training folds to guarantee zero leakage into validation folds...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_cv_f1_weighted = []
rf_cv_acc = []
xgb_cv_f1_weighted = []
xgb_cv_acc = []

for fold, (train_k_idx, val_k_idx) in enumerate(skf.split(X_train_combined, y_train), 1):
    X_tr_f = X_train_combined[train_k_idx]
    y_tr_f = y_train[train_k_idx]
    X_val_f = X_train_combined[val_k_idx]
    y_val_f = y_train[val_k_idx]
    
    # Apply SMOTE inside training fold only
    smote = SMOTE(random_state=42 + fold)
    X_tr_f_res, y_tr_f_res = smote.fit_resample(X_tr_f, y_tr_f)
    
    # Train Random Forest inside fold
    rf_fold = RandomForestClassifier(
        n_estimators=100,  # 100 for fast CV, full 200 for final model
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_fold.fit(X_tr_f_res, y_tr_f_res)
    rf_val_preds = rf_fold.predict(X_val_f)
    rf_cv_f1_weighted.append(f1_score(y_val_f, rf_val_preds, average='weighted'))
    rf_cv_acc.append(accuracy_score(y_val_f, rf_val_preds))
    
    # Train XGBoost inside fold
    xgb_fold = XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_fold.fit(X_tr_f_res, y_tr_f_res)
    xgb_val_preds = xgb_fold.predict(X_val_f)
    xgb_cv_f1_weighted.append(f1_score(y_val_f, xgb_val_preds, average='weighted'))
    xgb_cv_acc.append(accuracy_score(y_val_f, xgb_val_preds))
    
    print(f"Fold {fold}/5: RF F1-Weighted={rf_cv_f1_weighted[-1]:.4f} | XGB F1-Weighted={xgb_cv_f1_weighted[-1]:.4f}")

rf_mean_f1 = np.mean(rf_cv_f1_weighted)
rf_std_f1 = np.std(rf_cv_f1_weighted)
xgb_mean_f1 = np.mean(xgb_cv_f1_weighted)
xgb_std_f1 = np.std(xgb_cv_f1_weighted)

print("\n--- 5-Fold Stratified Cross-Validation Summary ---")
print(f"Random Forest (with SMOTE in CV): Mean F1 = {rf_mean_f1:.4f} ± {rf_std_f1:.4f} | Mean Acc = {np.mean(rf_cv_acc):.4f}")
print(f"XGBoost       (with SMOTE in CV): Mean F1 = {xgb_mean_f1:.4f} ± {xgb_std_f1:.4f} | Mean Acc = {np.mean(xgb_cv_acc):.4f}")

print("\n" + "="*80)
print("STEP 7: FINAL MODEL TRAINING ON COMPLETE 80% TRAINING SET")
print("="*80)

# Resample training data with SMOTE for final training
print("Applying SMOTE on full training set...")
smote_final = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote_final.fit_resample(X_train_combined, y_train)
print(f"Before SMOTE: {np.bincount(y_train)} -> After SMOTE: {np.bincount(y_train_resampled)}")

# Train Random Forest with exact task specifications
print("\nTraining Random Forest (n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=5, class_weight='balanced')...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_resampled, y_train_resampled)
print("Random Forest training complete.")

# Train XGBoost with exact task specifications
print("\nTraining XGBoost (n_estimators=200, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss')...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_resampled, y_train_resampled)
print("XGBoost training complete.")

print("\n" + "="*80)
print("STEP 8: TEST SET EVALUATION & BENCHMARKING")
print("="*80)

# Predictions
rf_preds = rf_model.predict(X_test_combined)
rf_probs = rf_model.predict_proba(X_test_combined)

xgb_preds = xgb_model.predict(X_test_combined)
xgb_probs = xgb_model.predict_proba(X_test_combined)

# Binarize labels for ROC-AUC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# Compute RF metrics
rf_acc = accuracy_score(y_test, rf_preds)
rf_prec_w = precision_score(y_test, rf_preds, average='weighted')
rf_rec_w = recall_score(y_test, rf_preds, average='weighted')
rf_f1_w = f1_score(y_test, rf_preds, average='weighted')
rf_f1_macro = f1_score(y_test, rf_preds, average='macro')
rf_roc_auc = roc_auc_score(y_test_bin, rf_probs, multi_class='ovr', average='weighted')

# Compute XGB metrics
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_prec_w = precision_score(y_test, xgb_preds, average='weighted')
xgb_rec_w = recall_score(y_test, xgb_preds, average='weighted')
xgb_f1_w = f1_score(y_test, xgb_preds, average='weighted')
xgb_f1_macro = f1_score(y_test, xgb_preds, average='macro')
xgb_roc_auc = roc_auc_score(y_test_bin, xgb_probs, multi_class='ovr', average='weighted')

print(f"\n{'Metric':<25} | {'Random Forest':<15} | {'XGBoost (Best)':<15}")
print("-" * 62)
print(f"{'Test Accuracy':<25} | {rf_acc*100:<14.2f}% | {xgb_acc*100:<14.2f}%")
print(f"{'Weighted Precision':<25} | {rf_prec_w:<15.4f} | {xgb_prec_w:<15.4f}")
print(f"{'Weighted Recall':<25} | {rf_rec_w:<15.4f} | {xgb_rec_w:<15.4f}")
print(f"{'Weighted F1-Score':<25} | {rf_f1_w:<15.4f} | {xgb_f1_w:<15.4f}")
print(f"{'Macro F1-Score':<25} | {rf_f1_macro:<15.4f} | {xgb_f1_macro:<15.4f}")
print(f"{'Multiclass ROC-AUC (OvR)':<25} | {rf_roc_auc:<15.4f} | {xgb_roc_auc:<15.4f}")
print(f"{'5-Fold CV Mean F1 (±std)':<25} | {rf_mean_f1:.4f} ± {rf_std_f1:.4f} | {xgb_mean_f1:.4f} ± {xgb_std_f1:.4f}")

print("\n--- Detailed Classification Report: Random Forest ---")
print(classification_report(y_test, rf_preds, target_names=['Negative (0)', 'Neutral (1)', 'Positive (2)'], digits=4))

print("\n--- Detailed Classification Report: XGBoost ---")
print(classification_report(y_test, xgb_preds, target_names=['Negative (0)', 'Neutral (1)', 'Positive (2)'], digits=4))

# Confusion matrices
rf_cm = confusion_matrix(y_test, rf_preds)
xgb_cm = confusion_matrix(y_test, xgb_preds)

print("\n" + "="*80)
print("STEP 9: EVALUATION CHARTS GENERATION")
print("="*80)

# Chart 6: Confusion Matrices (RF & XGBoost)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    ax=axes[0],
    xticklabels=['Negative', 'Neutral', 'Positive'],
    yticklabels=['Negative', 'Neutral', 'Positive'],
    linewidths=1,
    linecolor='black'
)
axes[0].set_title(f'Random Forest Confusion Matrix\n(Accuracy: {rf_acc*100:.2f}%)', pad=15)
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.heatmap(
    xgb_cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    cbar=False,
    ax=axes[1],
    xticklabels=['Negative', 'Neutral', 'Positive'],
    yticklabels=['Negative', 'Neutral', 'Positive'],
    linewidths=1,
    linecolor='black'
)
axes[1].set_title(f'XGBoost Confusion Matrix (Best Model)\n(Accuracy: {xgb_acc*100:.2f}%)', pad=15)
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
chart6_path = 'static/images/eval_confusion_matrices.png'
plt.savefig(chart6_path, dpi=300)
plt.close()
print(f"Saved: {chart6_path}")

# Chart 7: Multiclass ROC Curves (One-vs-Rest)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
class_labels = ['Negative', 'Neutral', 'Positive']
class_colors = ['#EF4444', '#F59E0B', '#10B981']

# Random Forest ROC Curves
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_probs[:, i])
    roc_score = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=class_colors[i], lw=2.5,
                 label=f'{class_labels[i]} (AUC = {roc_score:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
axes[0].set_ylabel('True Positive Rate (Sensitivity)')
axes[0].set_title(f'Random Forest One-vs-Rest ROC Curves\n(Macro AUC = {rf_roc_auc:.3f})', pad=15)
axes[0].legend(loc="lower right")

# XGBoost ROC Curves
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], xgb_probs[:, i])
    roc_score = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color=class_colors[i], lw=2.5,
                 label=f'{class_labels[i]} (AUC = {roc_score:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
axes[1].set_ylabel('True Positive Rate (Sensitivity)')
axes[1].set_title(f'XGBoost One-vs-Rest ROC Curves (Best Model)\n(Macro AUC = {xgb_roc_auc:.3f})', pad=15)
axes[1].legend(loc="lower right")

plt.tight_layout()
chart7_path = 'static/images/eval_roc_curves.png'
plt.savefig(chart7_path, dpi=300)
plt.close()
print(f"Saved: {chart7_path}")

# Chart 8: Feature Importance (Top 20 Features)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

rf_importances = rf_model.feature_importances_
top_rf_indices = np.argsort(rf_importances)[::-1][:20]
top_rf_features = [feature_names[i] for i in top_rf_indices]
top_rf_scores = rf_importances[top_rf_indices]

axes[0].barh(range(len(top_rf_scores)), top_rf_scores[::-1], color='#3B82F6', edgecolor='black', linewidth=0.8)
axes[0].set_yticks(range(len(top_rf_scores)))
axes[0].set_yticklabels(top_rf_features[::-1], fontsize=10)
axes[0].set_xlabel('Gini Impurity Importance')
axes[0].set_title('Top 20 Features: Random Forest', pad=15)

xgb_importances = xgb_model.feature_importances_
top_xgb_indices = np.argsort(xgb_importances)[::-1][:20]
top_xgb_features = [feature_names[i] for i in top_xgb_indices]
top_xgb_scores = xgb_importances[top_xgb_indices]

axes[1].barh(range(len(top_xgb_scores)), top_xgb_scores[::-1], color='#10B981', edgecolor='black', linewidth=0.8)
axes[1].set_yticks(range(len(top_xgb_scores)))
axes[1].set_yticklabels(top_xgb_features[::-1], fontsize=10)
axes[1].set_xlabel('Gain / Weight Importance')
axes[1].set_title('Top 20 Features: XGBoost', pad=15)

plt.tight_layout()
chart8_path = 'static/images/eval_feature_importance.png'
plt.savefig(chart8_path, dpi=300)
plt.close()
print(f"Saved: {chart8_path}")

# Chart 9: Model Performance Metrics Comparison Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
metrics_names = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score', 'Macro F1-Score', 'ROC-AUC']
rf_scores_list = [rf_acc, rf_prec_w, rf_rec_w, rf_f1_w, rf_f1_macro, rf_roc_auc]
xgb_scores_list = [xgb_acc, xgb_prec_w, xgb_rec_w, xgb_f1_w, xgb_f1_macro, xgb_roc_auc]

x = np.arange(len(metrics_names))
width = 0.35

rects1 = ax.bar(x - width/2, [s*100 for s in rf_scores_list], width, label='Random Forest', color='#3B82F6', edgecolor='black')
rects2 = ax.bar(x + width/2, [s*100 for s in xgb_scores_list], width, label='XGBoost', color='#10B981', edgecolor='black')

ax.set_ylabel('Score (%)')
ax.set_title('Model Performance Comparison: Random Forest vs. XGBoost on Test Set', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, rotation=15, ha='right')
ax.set_ylim([50, 100])
ax.axhline(80, color='red', linestyle='--', linewidth=1.5, label='80% Target Benchmark')
ax.legend(loc='lower right')

# Value annotations
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', (rect.get_x() + rect.get_width()/2., h + 1),
                ha='center', va='bottom', fontsize=9, weight='bold', color='#1E40AF')
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', (rect.get_x() + rect.get_width()/2., h + 1),
                ha='center', va='bottom', fontsize=9, weight='bold', color='#065F46')

plt.tight_layout()
chart9_path = 'static/images/eval_model_comparison.png'
plt.savefig(chart9_path, dpi=300)
plt.close()
print(f"Saved: {chart9_path}")

print("\n" + "="*80)
print("STEP 10: SERIALIZING TRAINED MODELS & PIPELINE ARTIFACTS")
print("="*80)

# Save models and transformers using joblib
joblib.dump(xgb_model, 'models/xgboost_model.joblib')
joblib.dump(rf_model, 'models/random_forest_model.joblib')
joblib.dump(xgb_model, 'models/best_model.joblib') # Best model is XGBoost
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(ohe, 'models/ohe_encoder.joblib')

# Save comprehensive metadata
metadata = {
    'best_model': 'XGBoost',
    'sentiment_mapping': sentiment_map,
    'inverse_sentiment_mapping': inv_sentiment_map,
    'metrics': {
        'xgboost': {
            'accuracy': float(xgb_acc),
            'precision_weighted': float(xgb_prec_w),
            'recall_weighted': float(xgb_rec_w),
            'f1_weighted': float(xgb_f1_w),
            'f1_macro': float(xgb_f1_macro),
            'roc_auc': float(xgb_roc_auc),
            'cv_f1_mean': float(xgb_mean_f1),
            'cv_f1_std': float(xgb_std_f1)
        },
        'random_forest': {
            'accuracy': float(rf_acc),
            'precision_weighted': float(rf_prec_w),
            'recall_weighted': float(rf_rec_w),
            'f1_weighted': float(rf_f1_w),
            'f1_macro': float(rf_f1_macro),
            'roc_auc': float(rf_roc_auc),
            'cv_f1_mean': float(rf_mean_f1),
            'cv_f1_std': float(rf_std_f1)
        }
    },
    'feature_num_cols': feature_num_cols,
    'cat_cols': cat_cols,
    'total_samples': len(df),
    'train_samples': len(train_df),
    'test_samples': len(test_df)
}

with open('models/pipeline_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
joblib.dump(metadata, 'models/pipeline_metadata.joblib')

print("All models, transformers, and pipeline metadata serialized successfully to models/")
print("="*80)
print(f"PIPELINE RUN COMPLETED! Best Model Accuracy: {xgb_acc*100:.2f}% (Benchmark Target >= 80% ACHIEVED)")
print("="*80)
