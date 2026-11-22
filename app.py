"""
AI-ML-068: Airline Customer Sentiment Analysis
Flask Web Application for Real-Time Sentiment Prediction & Batch CSV Upload
"""

import os
import io
import re
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
import joblib

import nltk
# Add local nltk_data to search path
local_nltk = os.path.abspath('nltk_data')
if local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, local_nltk)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)
app.secret_key = "aerosense-secret-key-ai-ml-068"

# Global model cache
MODELS = {}
TRANSFORMERS = {}
METADATA = {}
BATCH_CACHE = []

# Preprocessing helpers
base_sw = set(stopwords.words('english'))
negations = {'not', 'no', 'never', 'nor', 'neither', 'hardly', 'barely', 'scarcely', 
             'aint', 'arent', 'cant', 'couldnt', 'didnt', 'doesnt', 'dont', 'hadnt', 
             'hasnt', 'havent', 'isnt', 'mightnt', 'mustnt', 'neednt', 'shant', 
             'shouldnt', 'wasnt', 'werent', 'wont', 'wouldnt'}
custom_sw = base_sw - negations
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

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

def clean_tweet_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.lower().split()
    cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in custom_sw and len(w) > 1]
    return " ".join(cleaned_tokens)

def extract_sentiment_words(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    pos_cnt = sum(1 for w in words if w in pos_lexicon)
    neg_cnt = sum(1 for w in words if w in neg_lexicon)
    return pos_cnt, neg_cnt, pos_cnt - neg_cnt

def get_confidence_bucket(conf):
    """
    Calibrated confidence bucket segmentation:
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

ROBERTA_CACHE = {}

def get_roberta_pipeline():
    if 'model' not in ROBERTA_CACHE:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        tok = AutoTokenizer.from_pretrained(model_name)
        mod = AutoModelForSequenceClassification.from_pretrained(model_name)
        mod.eval()
        ROBERTA_CACHE['tokenizer'] = tok
        ROBERTA_CACHE['model'] = mod
    return ROBERTA_CACHE['tokenizer'], ROBERTA_CACHE['model']

def predict_with_roberta(text):
    tok, mod = get_roberta_pipeline()
    import torch
    inputs = tok(str(text), return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        logits = mod(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs

def load_artifacts():
    global MODELS, TRANSFORMERS, METADATA
    if not MODELS:
        try:
            MODELS['xgboost'] = joblib.load('models/xgboost_model.joblib')
            MODELS['random_forest'] = joblib.load('models/random_forest_model.joblib')
            MODELS['best'] = joblib.load('models/best_model.joblib')
            TRANSFORMERS['tfidf'] = joblib.load('models/tfidf_vectorizer.joblib')
            TRANSFORMERS['scaler'] = joblib.load('models/scaler.joblib')
            TRANSFORMERS['ohe'] = joblib.load('models/ohe_encoder.joblib')
            
            with open('models/pipeline_metadata.json', 'r') as f:
                METADATA = json.load(f)
            print("Production models and transformers loaded into memory.")
        except Exception as e:
            print(f"Warning: Models not loaded yet ({e}). Waiting for training to complete.")

def build_feature_matrix(df_input):
    load_artifacts()
    tfidf = TRANSFORMERS['tfidf']
    scaler = TRANSFORMERS['scaler']
    ohe = TRANSFORMERS['ohe']
    
    # 1. Cleaned text
    clean_texts = df_input['text'].apply(clean_tweet_text)
    X_tfidf = tfidf.transform(clean_texts)
    
    # 2. Structural & sentiment features
    df_input['text_length'] = df_input['text'].astype(str).str.len()
    df_input['word_count'] = df_input['text'].astype(str).apply(lambda x: len(x.split()))
    df_input['hashtag_count'] = df_input['text'].astype(str).apply(lambda x: len(re.findall(r'#\w+', x)))
    df_input['mention_count'] = df_input['text'].astype(str).apply(lambda x: len(re.findall(r'@\w+', x)))
    df_input['url_count'] = df_input['text'].astype(str).apply(lambda x: len(re.findall(r'https?://\S+|www\.\S+', x)))
    df_input['excl_count'] = df_input['text'].astype(str).apply(lambda x: x.count('!'))
    df_input['q_count'] = df_input['text'].astype(str).apply(lambda x: x.count('?'))
    df_input['caps_ratio'] = df_input['text'].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    
    lex = df_input['text'].apply(extract_sentiment_words)
    df_input['positive_word_count'] = [x[0] for x in lex]
    df_input['negative_word_count'] = [x[1] for x in lex]
    df_input['sentiment_score'] = [x[2] for x in lex]
    
    vader = df_input['text'].astype(str).apply(sia.polarity_scores)
    df_input['vader_compound'] = [x['compound'] for x in vader]
    df_input['vader_pos'] = [x['pos'] for x in vader]
    df_input['vader_neg'] = [x['neg'] for x in vader]
    df_input['vader_neu'] = [x['neu'] for x in vader]
    
    feature_num_cols = [
        'text_length', 'word_count', 'hashtag_count', 'mention_count', 'url_count',
        'excl_count', 'q_count', 'caps_ratio',
        'positive_word_count', 'negative_word_count', 'sentiment_score',
        'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu'
    ]
    X_num = scaler.transform(df_input[feature_num_cols])
    
    # 3. Categorical features
    if 'airline' not in df_input.columns:
        df_input['airline'] = 'Other'
    if 'confidence_bucket' not in df_input.columns:
        df_input['confidence_bucket'] = 'high'
        
    X_cat = ohe.transform(df_input[['airline', 'confidence_bucket']])
    
    # Combined sparse matrix
    X_combined = hstack([X_tfidf, csr_matrix(X_num), X_cat]).tocsr()
    return X_combined, clean_texts

@app.route('/')
def index():
    load_artifacts()
    return render_template('index.html', result=None, active_page='index')

@app.route('/predict', methods=['POST'])
def predict_single():
    load_artifacts()
    tweet_text = request.form.get('tweet_text', '').strip()
    airline = request.form.get('airline', 'Other')
    model_choice = request.form.get('model_choice', 'xgboost')
    
    if not tweet_text:
        return redirect(url_for('index'))
    
    # Auto-detect airline if 'Other'
    if airline == 'Other':
        t_low = tweet_text.lower()
        if 'united' in t_low: airline = 'United'
        elif 'american' in t_low or 'aa' in t_low: airline = 'American'
        elif 'delta' in t_low: airline = 'Delta'
        elif 'southwest' in t_low or 'southwestair' in t_low: airline = 'Southwest'
        elif 'usairways' in t_low or 'us airways' in t_low: airline = 'US Airways'
        elif 'virgin' in t_low or 'virginamerica' in t_low: airline = 'Virgin America'
    
    if model_choice == 'roberta':
        model_name_display = "Twitter-RoBERTa (Transformer)"
        pred_idx, probs = predict_with_roberta(tweet_text)
        input_df = pd.DataFrame([{'text': tweet_text, 'airline': airline, 'confidence_bucket': 'high'}])
        _, clean_texts = build_feature_matrix(input_df)
    else:
        input_df = pd.DataFrame([{'text': tweet_text, 'airline': airline, 'confidence_bucket': 'high'}])
        X_mat, clean_texts = build_feature_matrix(input_df)
        model = MODELS.get(model_choice, MODELS.get('best'))
        model_name_display = "XGBoost Classifier" if model_choice == 'xgboost' else "Random Forest Classifier"
        pred_idx = model.predict(X_mat)[0]
        probs = model.predict_proba(X_mat)[0]
    
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    pred_sentiment = sentiment_map[pred_idx]
    confidence = float(probs[pred_idx])
    
    # Feature dictionary for inspector
    pos_cnt, neg_cnt, sent_score = extract_sentiment_words(tweet_text)
    vader = sia.polarity_scores(tweet_text)
    
    result = {
        'sentiment': pred_sentiment,
        'confidence': confidence,
        'confidence_bucket': get_confidence_bucket(confidence),
        'model_used': model_name_display,
        'probabilities': {
            'negative': float(probs[0]),
            'neutral': float(probs[1]),
            'positive': float(probs[2])
        },
        'features': {
            'clean_text': clean_texts.iloc[0],
            'text_length': len(tweet_text),
            'word_count': len(tweet_text.split()),
            'mention_count': len(re.findall(r'@\w+', tweet_text)),
            'positive_words': pos_cnt,
            'negative_words': neg_cnt,
            'sentiment_score': sent_score,
            'vader_compound': float(vader['compound'])
        }
    }
    
    return render_template(
        'index.html',
        result=result,
        tweet_input=tweet_text,
        selected_airline=airline,
        selected_model=model_choice,
        active_page='index'
    )

@app.route('/batch')
def batch():
    load_artifacts()
    return render_template('batch.html', batch_summary=None, batch_results=None, active_page='batch')

@app.route('/batch_process', methods=['POST'])
def batch_process():
    global BATCH_CACHE
    load_artifacts()
    if 'csv_file' not in request.files:
        flash('No file selected', 'danger')
        return redirect(url_for('batch'))
        
    file = request.files['csv_file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('batch'))
        
    model_select = request.form.get('model_select', 'xgboost')
    if model_select == 'roberta':
        model_display = "Twitter-RoBERTa (Transformer)"
    else:
        model = MODELS.get(model_select, MODELS.get('best'))
        model_display = "XGBoost" if model_select == 'xgboost' else "Random Forest"
    
    try:
        try:
            df_batch = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df_batch = pd.read_csv(file, on_bad_lines='skip', encoding_errors='replace')
        if 'text' not in df_batch.columns:
            flash("Uploaded CSV must contain a 'text' column.", 'danger')
            return redirect(url_for('batch'))
            
        if 'airline' not in df_batch.columns:
            df_batch['airline'] = 'Other'
            
        df_batch['airline'] = df_batch['airline'].fillna('Other')
        df_batch['confidence_bucket'] = 'high'
        
        if model_select == 'roberta':
            tok, mod = get_roberta_pipeline()
            import torch
            all_probs = []
            for t in df_batch['text']:
                inputs = tok(str(t), return_tensors='pt', truncation=True, max_length=128)
                with torch.no_grad():
                    p = torch.softmax(mod(**inputs).logits, dim=1)[0].numpy()
                all_probs.append(p)
            probs = np.array(all_probs)
            preds = np.argmax(probs, axis=1)
        else:
            X_batch, _ = build_feature_matrix(df_batch.copy())
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch)
        
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        df_batch['predicted_sentiment'] = [sentiment_map[p] for p in preds]
        df_batch['confidence'] = [float(probs[i, preds[i]]) for i in range(len(preds))]
        df_batch['confidence_bucket'] = [get_confidence_bucket(c) for c in df_batch['confidence']]
        df_batch['prob_negative'] = probs[:, 0]
        df_batch['prob_neutral'] = probs[:, 1]
        df_batch['prob_positive'] = probs[:, 2]
        
        # Save to memory cache for download
        BATCH_CACHE = df_batch.copy()
        
        # Calculate summary statistics
        total = len(df_batch)
        neg_count = int((df_batch['predicted_sentiment'] == 'negative').sum())
        neu_count = int((df_batch['predicted_sentiment'] == 'neutral').sum())
        pos_count = int((df_batch['predicted_sentiment'] == 'positive').sum())
        high_conf_count = int((df_batch['confidence_bucket'] == 'high').sum())
        med_conf_count = int((df_batch['confidence_bucket'] == 'medium').sum())
        low_conf_count = int((df_batch['confidence_bucket'] == 'low').sum())
        
        batch_summary = {
            'total': total,
            'model_used': model_display,
            'negative_count': neg_count,
            'negative_pct': (neg_count / total * 100) if total > 0 else 0,
            'neutral_count': neu_count,
            'neutral_pct': (neu_count / total * 100) if total > 0 else 0,
            'positive_count': pos_count,
            'positive_pct': (pos_count / total * 100) if total > 0 else 0,
            'high_conf_count': high_conf_count,
            'med_conf_count': med_conf_count,
            'low_conf_count': low_conf_count
        }
        
        results_list = []
        for _, row in df_batch.iterrows():
            results_list.append({
                'airline': row.get('airline', 'Other'),
                'text': row['text'],
                'sentiment': row['predicted_sentiment'],
                'confidence': row['confidence'],
                'confidence_bucket': row['confidence_bucket'],
                'prob_negative': row['prob_negative'],
                'prob_neutral': row['prob_neutral'],
                'prob_positive': row['prob_positive']
            })
            
        return render_template(
            'batch.html',
            batch_summary=batch_summary,
            batch_results=results_list,
            active_page='batch'
        )
    except Exception as e:
        flash(f"Error processing CSV: {str(e)}", 'danger')
        return redirect(url_for('batch'))

@app.route('/export_batch')
def export_batch_results():
    global BATCH_CACHE
    if BATCH_CACHE is None or len(BATCH_CACHE) == 0:
        return redirect(url_for('batch'))
        
    out = io.StringIO()
    BATCH_CACHE.to_csv(out, index=False)
    mem = io.BytesIO()
    mem.write(out.getvalue().encode('utf-8'))
    mem.seek(0)
    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name='aerosense_scored_tweets.csv'
    )

@app.route('/download_sample')
def download_sample_csv():
    sample_file = 'sample_test_tweets.csv'
    if os.path.exists(sample_file):
        return send_file(sample_file, as_attachment=True, download_name='sample_airline_tweets.csv')
    return redirect(url_for('batch'))

@app.route('/dashboard')
def dashboard():
    load_artifacts()
    return render_template('dashboard.html', metadata=METADATA, active_page='dashboard')

# JSON API Endpoint for REST clients
@app.route('/api/predict', methods=['POST'])
def api_predict():
    load_artifacts()
    data = request.get_json(force=True)
    text = data.get('text', '')
    airline = data.get('airline', 'Other')
    model_choice = data.get('model', 'xgboost')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
        
    if model_choice == 'roberta':
        pred_idx, probs = predict_with_roberta(text)
    else:
        input_df = pd.DataFrame([{'text': text, 'airline': airline, 'confidence_bucket': 'high'}])
        X_mat, _ = build_feature_matrix(input_df)
        model = MODELS.get(model_choice, MODELS.get('best'))
        pred_idx = model.predict(X_mat)[0]
        probs = model.predict_proba(X_mat)[0]
    
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    conf = float(probs[pred_idx])
    return jsonify({
        'predicted_sentiment': sentiment_map[pred_idx],
        'sentiment': sentiment_map[pred_idx],
        'confidence': conf,
        'confidence_bucket': get_confidence_bucket(conf),
        'prob_negative': float(probs[0]),
        'prob_neutral': float(probs[1]),
        'prob_positive': float(probs[2]),
        'probabilities': {
            'negative': float(probs[0]),
            'neutral': float(probs[1]),
            'positive': float(probs[2])
        }
    })

if __name__ == '__main__':
    load_artifacts()
    app.run(host='127.0.0.1', port=5000, debug=False)
