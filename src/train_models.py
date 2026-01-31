"""
Model Training Module (Phase 3)

Trains classifiers:
1. Logistic Regression
2. Random Forest
3. MLP Classifier (Neural Head on S-BERT features)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import time

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.npz"
LABELS_PATH = PROCESSED_DATA_DIR / "labels.npy"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

class ModelTrainer:
    def __init__(self):
        # Using Pipelines to ensure features are scaled
        self.models = {
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
            ]),
            'random_forest': Pipeline([
                # Random Forest doesn't strictly need scaling, but it doesn't hurt when mixing features
                ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced'))
            ]),
            'distilbert_head': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, activation='relu'))
            ])
        }
        
    def load_data(self):
        print("📂 Loading features and labels...")
        data = np.load(FEATURES_PATH)
        X = data['features']
        y = np.load(LABELS_PATH)
        
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        
        # Check for NaNs or Infs
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("⚠️  Warning: Features contain NaNs or Infs. Cleaning...")
            X = np.nan_to_num(X)
            
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train(self):
        print("="*60)
        print("PHASE 3: MODEL TRAINING")
        print("="*60)
        
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Save test set for Phase 4
        print("💾 Saving test set for Phase 4...")
        np.savez_compressed(PROCESSED_DATA_DIR / "test_data.npz", X=X_test, y=y_test)
        
        for name, model in self.models.items():
            print(f"\n🏋️‍♂️ Training {name}...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            elapsed = time.time() - start_time
            print(f"   ✅ Trained in {elapsed:.2f}s")
            
            # Quick validation
            score = model.score(X_test, y_test)
            print(f"   🎯 Accuracy: {score:.4f}")
            
            # Save model
            model_path = MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)
            print(f"   💾 Saved to {model_path}")
            
        print("\n✅ PHASE 3 COMPLETE!")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
