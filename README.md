# AeroSense AI: Airline Customer Sentiment Analysis Using Machine Learning & Transformers (AI-ML-068)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Hist%20Gradient%20Boost-green.svg)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![DirectML](https://img.shields.io/badge/AMD%20Radeon-DirectML%20Accelerated-ED1C24.svg)](https://microsoft.github.io/DirectML/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 📌 Project Overview

**AeroSense AI** is an enterprise-grade Natural Language Processing (NLP) and Machine Learning system that classifies commercial airline customer tweets into ternary sentiment categories: **Negative (0)**, **Neutral (1)**, and **Positive (2)**.

Trained and evaluated on the **Twitter US Airline Sentiment Dataset** (14,640 records across 6 major US carriers: *United Airlines, American Airlines, US Airways, Southwest Airlines, Delta Air Lines, and Virgin America*), AeroSense AI overcomes severe class imbalance (62.7% Negative, 21.2% Neutral, 16.1% Positive) and resolves the classical **Neutral-to-Negative misclassification bottleneck** using a Twitter-pretrained Transformer backbone (`cardiffnlp/twitter-roberta-base-sentiment-latest`) and a hybrid stacking meta-classifier.

---

## 🚀 Key Architectural Innovations

1. **Zero-Leakage In-Fold SMOTE Cross-Validation**:
   - Synthetic Minority Over-sampling Technique (SMOTE) is applied strictly within the training folds of 5-fold Stratified Cross-Validation (`86.89% ± 0.17%` test accuracy), guaranteeing zero data leakage into evaluation splits.
2. **Domain-Specific NLP & Sentiment Features**:
   - 15 engineered structural, punctuation, and aviation lexicon features combined with a 5,000-feature bi-gram TF-IDF matrix using custom negation-preserving stopwords (*'not'*, *'never'*, *'no'* retained to prevent polarity inversion).
3. **CardiffNLP Twitter-RoBERTa Transformer**:
   - Pretrained on 124M tweets and fine-tuned with class-weighted cross-entropy (`weights = [0.53, 1.57, 2.06]`) and label smoothing (`0.05`) to boost minority Neutral recall without compromising Negative/Positive precision.
4. **Hybrid Stacking Meta-Classifier & Multiplier Tuning**:
   - Out-of-fold transformer probability margins and prediction entropy are fused with tabular domain features into an XGBoost meta-learner, combined with optimal post-processing multipliers `[1.00, 2.35, 0.80]` that reduce Neutral-to-Negative errors by 64 to 789 tweets.
5. **Calibrated Tripartite Confidence Segmentation**:
   - Replaces naive confidence tagging with calibrated thresholds: **High (>0.85, 89.24% acc)**, **Medium (0.65–0.85, 63.05% acc)**, and **Low (<0.65, 45.45% acc)** for automated enterprise CRM routing.
6. **AMD Radeon DirectML Hardware Acceleration**:
   - Verified on **AMD Radeon(TM) 610M** APU via Microsoft DirectML (`onnxruntime-directml`), with automated AVX2 multi-threaded CPU fallback.
7. **Production Flask Web Application**:
   - Real-time single-tweet scoring, batch CSV upload, interactive KPI dashboards, scored CSV export, and RESTful JSON API (`POST /api/predict`).

---

## 📊 Benchmark Results

| Model Architecture | Accuracy | Macro F1 | Negative F1 | Neutral Recall | Neutral F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Balanced)** | 75.44% | 0.6920 | 0.8250 | 64.20% | 0.5840 | 0.884 |
| **Baseline XGBoost** | 86.89% | 0.8291 | 0.9080 | 53.23% (789 misclass) | 0.7250 | 0.912 |
| **Twitter-RoBERTa + Stacking (All)** | **87.65%** | **0.8482** | **0.9150** | **67.74% (+14.5%)** | **0.7810** | **0.941** |
| **High-Confidence Predictions (>0.85)** | **89.24%** | **0.8640** | **0.9380** | **78.20%** | **0.8240** | **0.962** |
| **Ground-Truth Subset (`conf == 1.0`)** | **86.82%** | **0.8181** | **0.9163** | **71.20%** | **0.6966** | **0.948** |

---

## 📁 Repository Structure

```
.
├── AeroSense_AI_Airline_Sentiment_Analysis_Project_Report.pdf   # 30-page Publication Report (PDF)
├── AeroSense_AI_Airline_Sentiment_Analysis_Project_Report.docx  # Publication Report (DOCX)
├── AI-ML-068_Airline_Sentiment_Analysis_Complete_Project.zip    # Final Archive (Report + Code)
├── source_code.zip                                             # Source Code Package
├── PROJECT_REPORT.md                                           # Full Technical Report (Markdown)
├── README.md                                                   # Project Documentation
├── requirements.txt                                            # Python Dependencies
├── Tweets.csv                                                  # Benchmark Twitter US Airline Dataset
├── sample_test_tweets.csv                                      # 15-Sample Batch Test Tweets
├── app.py                                                      # Flask Web App & REST API Engine
├── eda_and_train.py                                            # EDA, Feature Engineering & Baseline Training
├── upgrade_pipeline.py                                         # RoBERTa Fine-Tuning & GPU/DirectML Pipeline
├── test_stacking_and_thresholds.py                             # Stacking Meta-Classifier & Multipliers
├── run_full_verification.py                                    # Automated 10-Test Verification Harness
├── directml_test.py                                            # AMD Radeon DirectML Diagnostic Script
├── build_complete_report.py                                    # 30-Page Report Generator (PDF + DOCX)
├── models/                                                     # Serialized Model Artifacts
│   ├── best_model.joblib
│   ├── xgboost_model.joblib
│   ├── random_forest_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── scaler.joblib
│   ├── ohe_encoder.joblib
│   └── pipeline_metadata.json
├── templates/                                                  # HTML UI Templates (Bootstrap 5)
│   ├── base.html
│   ├── index.html
│   ├── batch.html
│   └── dashboard.html
└── static/                                                     # Visualizations & Styling
    ├── css/style.css
    └── images/                                                 # 15 High-Resolution Charts
        ├── confidence_bucket_distribution.png
        ├── eda_airline_sentiment_breakdown.png
        ├── eda_correlation_heatmap.png
        ├── eda_sentiment_distribution.png
        ├── eda_sentiment_score_analysis.png
        ├── eda_text_length_boxplot.png
        ├── eval_confusion_matrices.png
        ├── eval_feature_importance.png
        ├── eval_model_comparison.png
        ├── eval_neutral_recall_boost.png
        ├── eval_roc_curves.png
        ├── eval_upgrade_confusion_matrix.png
        ├── system_architecture_diagram.png
        ├── ui_screenshot_batch_upload.png
        └── ui_screenshot_single_tweet.png
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/nk1356330-ops/AI-ML-68.git
cd AI-ML-68
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage Guide

### 1. Run the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.
- **Single-Tweet Analyzer**: Real-time sentiment prediction, probability breakdown progress bars, and calibrated confidence badges.
- **Batch CSV Processor**: Upload CSV files (e.g., `sample_test_tweets.csv`) to score multiple tweets with summary analytics and exportable results.

### 2. RESTful JSON API
Send a `POST` request to `/api/predict`:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "@SouthwestAir loving the fast inflight wifi on flight WN302!", "model": "roberta"}'
```

**Response:**
```json
{
  "status": "success",
  "tweet": "@SouthwestAir loving the fast inflight wifi on flight WN302!",
  "predicted_sentiment": "positive",
  "confidence": 0.9412,
  "confidence_bucket": "high",
  "probabilities": {
    "negative": 0.0123,
    "neutral": 0.0465,
    "positive": 0.9412
  },
  "model_used": "roberta"
}
```

### 3. Execute Automated Verification Suite
Run the full 10-test unit and integration test harness:
```bash
python run_full_verification.py
```
Expected result: **`10 OUT OF 10 TESTS PASSED (100% Pass Rate)`**.

### 4. Retrain Baseline Models & Generate Visualizations
```bash
python eda_and_train.py
```

### 5. Generate the 30-Page Publication Report (PDF & DOCX)
```bash
python build_complete_report.py
```

---

## 📦 Deliverables Summary

- **PDF Project Report**: `AeroSense_AI_Airline_Sentiment_Analysis_Project_Report.pdf` (Strictly 30 pages, all 15 figures embedded, dataset link at end).
- **DOCX Project Report**: `AeroSense_AI_Airline_Sentiment_Analysis_Project_Report.docx`
- **Source Code Archive**: `source_code.zip` (All Python scripts, templates, static assets, requirements).
- **Master Final Archive**: `AI-ML-068_Airline_Sentiment_Analysis_Complete_Project.zip` (Report + Source Code).

---

## 🔗 Dataset Source & Attribution

- **Dataset Title**: Twitter US Airline Sentiment Dataset
- **Provider**: Crowdflower "Data for Everyone" Library
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Official Repository Link**: [https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
