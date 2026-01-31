"""
Guardrail Demo Module (Phase 6)

Real-time checking pipeline:
User Prompt -> Feature Extraction -> Classifier -> Allow/Block
"""

import numpy as np
import joblib
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "distilbert_head.joblib"

class SecurityGuardrail:
    def __init__(self):
        print("🛡️  Initializing Guardrail...")
        
        # Load Model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            
        print("   Loading classifier...")
        self.classifier = joblib.load(MODEL_PATH)
        
        # Load S-BERT
        print("   Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define Keywords (MUST match training!)
        self.keywords = [
            'ignore', 'override', 'system', 'developer', 'model', 'prior',
            'jailbreak', 'mode', 'unfiltered', 'dan', 'roleplay', 'simul'
        ]
        print("✅ Guardrail Ready")

    def _extract_features(self, text: str) -> np.ndarray:
        """Replicate feature extraction logic"""
        # Lexical
        char_len = len(text)
        word_len = len(text.split())
        caps_ratio = sum(1 for c in text if c.isupper()) / char_len if char_len > 0 else 0
        special_chars = sum(1 for c in text if c in '!?{}[]@#$%')
        punct_density = special_chars / char_len if char_len > 0 else 0
        text_lower = text.lower()
        keyword_flags = [1 if kw in text_lower else 0 for kw in self.keywords]
        
        lexical = [char_len, word_len, caps_ratio, punct_density] + keyword_flags
        
        # Structural
        role_switch_count = len(re.findall(r'(User:|Assistant:|System:|Human:|AI:)', text, re.IGNORECASE))
        stacking_count = text.count('\n-') + text.count('\n*') + text.count('1.') 
        nested_count = len(re.findall(r'(\{|\[|<|`{3})', text))
        
        structural = [role_switch_count, stacking_count, nested_count]
        
        # Semantic
        embedding = self.embedding_model.encode([text])[0]
        
        # Combine
        return np.hstack((lexical, structural, embedding)).reshape(1, -1)

    def scan(self, prompt: str):
        """Scan a prompt and return decision"""
        features = self._extract_features(prompt)
        
        # Predict
        prediction = self.classifier.predict(features)[0]
        probability = self.classifier.predict_proba(features)[0]
        
        # Map label
        labels = {0: 'BENIGN', 1: 'INJECTION', 2: 'JAILBREAK'}
        result = labels.get(prediction, "UNKNOWN")
        confidence = probability[prediction]
        
        return {
            'status': 'BLOCK' if prediction > 0 else 'ALLOW',
            'label': result,
            'confidence': float(confidence),
            'prompt_snippet': prompt[:50] + "..." if len(prompt) > 50 else prompt
        }

def interactive_demo():
    print("\n" + "="*50)
    print("🔒 AI PROMPT SECURITY GUARDRAIL DEMO")
    print("="*50)
    print("Type a prompt to scan (or 'exit' to quit)\n")
    
    guard = SecurityGuardrail()
    
    while True:
        user_input = input("\n📝 Enter prompt: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        decision = guard.scan(user_input)
        
        color = "\033[92m" if decision['status'] == 'ALLOW' else "\033[91m"
        reset = "\033[0m"
        
        print(f"\nDecision: {color}{decision['status']}{reset}")
        print(f"Type:     {decision['label']}")
        print(f"Score:    {decision['confidence']:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    interactive_demo()
