"""
Error Analysis Module (Phase 5)

Identifies and saves:
- False Positives (Benign -> Malicious)
- False Negatives (Malicious -> Benign)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.npz"
DATA_PATH = PROCESSED_DATA_DIR / "prompts.csv"

class ErrorAnalyzer:
    def __init__(self):
        self.model = None
        self.X_test = None
        self.y_test = None
        # We need original text to make sense of errors
        # Note: recovering original text aligned with test set is tricky unless we kept indices
        # For simplicity, we'll reload the full dataset and check predictions on it 
        # (or just a subset to find *some* errors)
    
    def run(self):
        print("="*60)
        print("PHASE 5: ERROR ANALYSIS")
        print("="*60)
        
        # Load best model (defaulting to Random Forest)
        model_path = MODELS_DIR / "random_forest.joblib"
        if not model_path.exists():
            print("❌ Model not found")
            return
            
        print(f"Loading {model_path.name}...")
        self.model = joblib.load(model_path)
        
        # Load original data + features
        # We process the whole dataset to find specific errors with text
        print("Loading full dataset for inspection...")
        df = pd.read_csv(DATA_PATH)
        data = np.load(PROCESSED_DATA_DIR / "features.npz")
        X = data['features']
        y = np.load(PROCESSED_DATA_DIR / "labels.npy")
        
        # Checking inputs
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X)
            
        print("Predicting...")
        y_pred = self.model.predict(X)
        
        df['pred'] = y_pred
        
        # 1. False Positives (Truth=0, Pred!=0)
        fps = df[(df['label'] == 0) & (df['pred'] != 0)].copy()
        print(f"Found {len(fps)} False Positives")
        
        # 2. False Negatives (Truth!=0, Pred=0)
        fns = df[(df['label'] != 0) & (df['pred'] == 0)].copy()
        print(f"Found {len(fns)} False Negatives")
        
        # Save samples
        output_path = RESULTS_DIR / "error_analysis.txt"
        with open(output_path, 'w') as f:
            f.write("ERROR ANALYSIS REPORT\n")
            f.write("=====================\n\n")
            
            f.write(f"FALSE POSITIVES (Benign classified as Malicious) - {len(fps)}\n")
            f.write("-" * 50 + "\n")
            for i, row in fps.head(50).iterrows():
                f.write(f"Confidence: N/A\n") # RF proba could be added
                f.write(f"Predicted: {row['pred']}, True: {row['label']}\n")
                f.write(f"Text: {row['text'][:500]}\n\n")
            
            f.write("\n\n" + "="*50 + "\n\n")
            
            f.write(f"FALSE NEGATIVES (Malicious classified as Benign) - {len(fns)}\n")
            f.write("-" * 50 + "\n")
            for i, row in fns.head(50).iterrows():
                f.write(f"Predicted: {row['pred']}, True: {row['label']}\n")
                f.write(f"Text: {row['text'][:500]}\n\n")
                
        print(f"✅ Saved analysis to {output_path}")

if __name__ == "__main__":
    analyzer = ErrorAnalyzer()
    analyzer.run()
