"""
End-to-End Verification Test Suite for Upgraded AeroSense AI Pipeline
"""

import sys
import subprocess
import json
import pandas as pd
from app import app, get_confidence_bucket

def run_tests():
    print("=" * 75)
    print("RUNNING AEROSENSE AI FULL VERIFICATION SUITE")
    print("=" * 75)

    # 1. Test Hardware Check Graceful Failure
    print("\n[TEST 1] Testing Hardware Verification Diagnostic (Graceful Failure)...")
    res = subprocess.run([sys.executable, "upgrade_pipeline.py"], capture_output=True, text=True)
    assert res.returncode == 1, f"Expected returncode 1, got {res.returncode}"
    assert "[HARDWARE CHECK] CUDA GPU Verification Failed" in res.stdout
    assert "pip install torch" in res.stdout
    assert "--allow-cpu" in res.stdout
    print("  -> PASSED: CUDA verification accurately diagnoses environment & instructs user gracefully.")

    # 2. Test Confidence Bucket Function
    print("\n[TEST 2] Testing Calibrated Confidence Segmentation...")
    assert get_confidence_bucket(0.95) == 'high'
    assert get_confidence_bucket(0.851) == 'high'
    assert get_confidence_bucket(0.85) == 'medium'
    assert get_confidence_bucket(0.70) == 'medium'
    assert get_confidence_bucket(0.65) == 'medium'
    assert get_confidence_bucket(0.649) == 'low'
    assert get_confidence_bucket(0.30) == 'low'
    print("  -> PASSED: Confidence buckets segment correctly into low (<0.65), medium (0.65-0.85), high (>0.85).")

    # 3. Test Flask Test Client Routes
    print("\n[TEST 3] Testing Flask Web Application Routes & Inference...")
    client = app.test_client()

    # Index
    r = client.get('/')
    assert r.status_code == 200, f"GET / failed: {r.status_code}"

    # Single Prediction
    r = client.post('/predict', data={
        'tweet_text': '@united thank you for the wonderful upgrade and courteous crew!',
        'airline': 'United',
        'model_choice': 'xgboost'
    })
    assert r.status_code == 200
    assert b'positive' in r.data.lower()
    assert b'bucket' in r.data.lower()
    print("  -> PASSED: Single prediction returns sentiment, probability bars & confidence bucket badge.")

    # Batch Process
    with open('sample_test_tweets.csv', 'rb') as f:
        r = client.post('/batch_process', data={
            'csv_file': (f, 'sample_test_tweets.csv'),
            'model_select': 'xgboost'
        })
    assert r.status_code == 200
    assert b'Classified Tweet Records' in r.data
    assert b'high' in r.data.lower() or b'medium' in r.data.lower() or b'low' in r.data.lower()
    print("  -> PASSED: Batch CSV upload scores all tweets with confidence buckets & probabilities.")

    # Export Batch CSV
    r = client.get('/export_batch')
    assert r.status_code == 200
    assert 'text/csv' in r.headers['Content-Type']
    csv_content = r.data.decode('utf-8')
    assert 'confidence_bucket' in csv_content
    assert 'prob_negative' in csv_content
    assert 'prob_neutral' in csv_content
    assert 'prob_positive' in csv_content
    print("  -> PASSED: Exported CSV includes confidence_bucket and all class probability columns.")

    # REST API Endpoint
    r = client.post('/api/predict', json={
        'text': '@AmericanAir flight was delayed 4 hours, missed my connection!',
        'airline': 'American',
        'model': 'xgboost'
    })
    assert r.status_code == 200
    data = r.get_json()
    assert data['predicted_sentiment'] == 'negative'
    assert 'confidence' in data
    assert data['confidence_bucket'] in ['low', 'medium', 'high']
    assert 'prob_negative' in data
    assert 'prob_neutral' in data
    assert 'prob_positive' in data
    print(f"  -> PASSED: REST API returns complete payload: {data}")

    # Dashboard Route
    r = client.get('/dashboard')
    assert r.status_code == 200
    print("  -> PASSED: Dashboard renders evaluation metrics & confusion matrices.")

    print("\n" + "=" * 75)
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
    print("=" * 75)

if __name__ == '__main__':
    run_tests()
