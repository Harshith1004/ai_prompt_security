"""
Feature Extraction Module (Phase 2)

Extracts three types of features:
1. Lexical: Length, punctuation, keywords
2. Structural: Role switching, instruction stacking, nested commands
3. Semantic: Sentence-BERT embeddings
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import joblib

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROCESSED_DATA_DIR / "prompts.csv"
OUTPUT_FEATURES = PROCESSED_DATA_DIR / "features.npz"
OUTPUT_LABELS = PROCESSED_DATA_DIR / "labels.npy"

# Ensure models dir exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class FeatureExtractor:
    def __init__(self):
        # Lexical Keywords
        self.keywords = [
            'ignore', 'override', 'system', 'developer', 'model', 'prior',
            'jailbreak', 'mode', 'unfiltered', 'dan', 'roleplay', 'simul'
        ]
        
        # S-BERT Model
        print("⏳ Loading Sentence-BERT model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded")

    def extract_lexical(self, text: str) -> List[float]:
        """Extract lexical features (length, punctuation, keywords)"""
        if not isinstance(text, str): return [0] * (4 + len(self.keywords))
        
        # 1. Length
        char_len = len(text)
        word_len = len(text.split())
        
        # 2. Capitalization ratio
        caps_ratio = sum(1 for c in text if c.isupper()) / char_len if char_len > 0 else 0
        
        # 3. Punctuation density
        special_chars = sum(1 for c in text if c in '!?{}[]@#$%')
        punct_density = special_chars / char_len if char_len > 0 else 0
        
        # 4. Keywords (Binary flags)
        text_lower = text.lower()
        keyword_flags = [1 if kw in text_lower else 0 for kw in self.keywords]
        
        return [char_len, word_len, caps_ratio, punct_density] + keyword_flags

    def extract_structural(self, text: str) -> List[float]:
        """Extract structural features (patterns, nesting)"""
        if not isinstance(text, str): return [0, 0, 0]
        
        # 1. Role Switching ("User:", "Assistant:")
        role_switch_count = len(re.findall(r'(User:|Assistant:|System:|Human:|AI:)', text, re.IGNORECASE))
        
        # 2. Instruction Stacking (Multiple bullet points or imperatives)
        stacking_count = text.count('\n-') + text.count('\n*') + text.count('1.') 
        
        # 3. Nested Commands (Code-like structures)
        nested_count = len(re.findall(r'(\{|\[|<|`{3})', text))
        
        return [role_switch_count, stacking_count, nested_count]

    def extract_semantic(self, texts: List[str]) -> np.ndarray:
        """Extract semantic embeddings"""
        print("⏳ Generating S-BERT embeddings (this may take a while)...")
        # Batch size 32 for efficiency
        embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
        return embeddings

    def process(self):
        print("="*60)
        print("PHASE 2: FEATURE ENGINEERING")
        print("="*60)
        
        # Load Data
        print(f"📂 Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        # df = df.sample(5000) # DEBUG: Uncomment to test quickly
        
        texts = df['text'].fillna("").astype(str).tolist()
        labels = df['label'].values
        
        # 1. Hand-crafted Features (Lexical + Structural)
        print("\n⚙️  Extracting Lexical & Structural features...")
        hand_crafted = []
        for text in tqdm(texts):
            lex = self.extract_lexical(text)
            struc = self.extract_structural(text)
            hand_crafted.append(lex + struc)
        
        X_handcrafted = np.array(hand_crafted)
        print(f"   Shape: {X_handcrafted.shape}")
        
        # 2. Semantic Features (Embeddings)
        X_semantic = self.extract_semantic(texts)
        print(f"   Shape: {X_semantic.shape}")
        
        # 3. Combine
        print("\n🔗 Combining features...")
        X_final = np.hstack((X_handcrafted, X_semantic))
        print(f"   Final Feature Matrix Shape: {X_final.shape}")
        
        # 4. Save
        print(f"\n💾 Saving to {OUTPUT_FEATURES}...")
        np.savez_compressed(OUTPUT_FEATURES, features=X_final)
        np.save(OUTPUT_LABELS, labels)
        
        # Save feature names for potential analysis
        feature_names = (
            ['char_len', 'word_len', 'caps_ratio', 'punct_density'] + 
            [f'kw_{k}' for k in self.keywords] +
            ['role_switches', 'stacking', 'nested_cmds'] +
            [f'emb_{i}' for i in range(X_semantic.shape[1])]
        )
        joblib.dump(feature_names, PROCESSED_DATA_DIR / "feature_names.pkl")
        
        print("\n✅ PHASE 2 COMPLETE!")

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.process()
