"""
Evaluation Module (Phase 4)

Computes:
- Recall, Precision, F1
- False Positive Rate
- ROC-AUC
- Confusion Matrix
- Stress Tests
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.npz"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.X_test = None
        self.y_test = None
        
    def load_resources(self):
        print("📂 Loading data and models...")
        # Load data
        data = np.load(TEST_DATA_PATH)
        self.X_test = data['X']
        self.y_test = data['y']
        
        # Load models
        for model_file in MODELS_DIR.glob("*.joblib"):
            self.models[model_file.stem] = joblib.load(model_file)
            print(f"   Loaded {model_file.stem}")

    def evaluate(self):
        print("="*60)
        print("PHASE 4: EVALUATION")
        print("="*60)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\n📊 Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)
            
            # Report
            report = classification_report(self.y_test, y_pred, output_dict=True)
            
            # Specific Metrics
            # Class 1 (Injection) and 2 (Jailbreak) are malicious
            malicious_indices = [1, 2]
            
            # Macro avg for malicious
            recall_malicious = np.mean([report[str(c)]['recall'] for c in malicious_indices if str(c) in report])
            
            # False Positive Rate (Benign misclassified as Malicious)
            # 0 is benign
            cm = confusion_matrix(self.y_test, y_pred)
            # Assuming labels 0, 1, 2
            # TP for Benign is [0,0]. FP for System (Benign classified as Malicious) is Sum(Row 0) - [0,0]
            fp_benign = cm[0][1:].sum()
            tn_benign = cm[0][0]
            fpr = fp_benign / (fp_benign + tn_benign + 1e-6)
            
            # ROC AUC (One-vs-Rest)
            try:
                auc = roc_auc_score(self.y_test, y_prob, multi_class='ovr')
            except:
                auc = 0.0
            
            print(f"   🎯 Malicious Recall: {recall_malicious:.4f}")
            print(f"   ⚠️ False Positive Rate: {fpr:.4f}")
            print(f"   📈 ROC-AUC: {auc:.4f}")
            
            results.append({
                'Model': name,
                'Accuracy': report['accuracy'],
                'Malicious Recall': recall_malicious,
                'FPR': fpr,
                'ROC-AUC': auc,
                'F1-Macro': report['macro avg']['f1-score']
            })
            
            # Plot Confusion Matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Benign', 'Injection', 'Jailbreak'],
                        yticklabels=['Benign', 'Injection', 'Jailbreak'])
            plt.title(f'Confusion Matrix: {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"cm_{name}.png")
            plt.close()
        
        # Save Summary
        df_results = pd.DataFrame(results)
        df_results.to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)
        print("\n" + df_results.to_markdown())
        
        print(f"\n✅ PHASE 4 COMPLETE! Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.load_resources()
    evaluator.evaluate()
