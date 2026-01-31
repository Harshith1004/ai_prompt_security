"""
Data Cleaning and Labeling Module

This script:
1. Loads all raw datasets
2. Assigns labels (0=benign, 1=injection, 2=jailbreak)
3. Cleans text (lowercase, remove duplicates, remove artifacts)
4. Saves to processed/prompts.csv
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


class DataCleaner:
    """Handles data cleaning and labeling operations"""
    
    def __init__(self):
        self.all_prompts = []
    
    def clean_text(self, text: str) -> str:
        """
        Clean individual text samples
        
        Cleaning steps:
        - Lowercase
        - Remove URLs
        - Remove HTML tags
        - Remove emojis
        - Strip extra whitespace
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove emojis (basic pattern)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def load_malicious_datasets(self) -> pd.DataFrame:
        """Load and label all malicious datasets"""
        
        print("\n🔴 Loading MALICIOUS datasets...")
        print("-" * 60)
        
        malicious_prompts = []
        
        # 1. Prompt Injections (label = 1)
        injection_path = RAW_DATA_DIR / "malicious" / "prompt_injections_deepset.csv"
        if injection_path.exists():
            df = pd.read_csv(injection_path)
            print(f"  ✓ Prompt Injections: {len(df):,} rows")
            
            # Extract text column (adjust column name as needed)
            if 'text' in df.columns:
                texts = df['text'].tolist()
            elif 'prompt' in df.columns:
                texts = df['prompt'].tolist()
            else:
                texts = df.iloc[:, 0].tolist()  # First column
            
            for text in texts:
                malicious_prompts.append({
                    'text': text,
                    'label': 1,  # prompt_injection
                    'source': 'deepset_injections'
                })
        
        # 2. Jailbreak Prompts (label = 2)
        jailbreak_path = RAW_DATA_DIR / "malicious" / "jailbreak_prompts.csv"
        if jailbreak_path.exists():
            df = pd.read_csv(jailbreak_path)
            print(f"  ✓ Jailbreak Prompts: {len(df):,} rows")
            
            # Extract prompt text
            if 'Prompt' in df.columns:
                texts = df['Prompt'].tolist()
            elif 'text' in df.columns:
                texts = df['text'].tolist()
            elif 'prompt' in df.columns:
                texts = df['prompt'].tolist()
            else:
                texts = df.iloc[:, 0].tolist()
            
            for text in texts:
                malicious_prompts.append({
                    'text': text,
                    'label': 2,  # jailbreak
                    'source': 'jailbreak_dataset'
                })
        
        # 3. Anthropic Red Team (label = 2, treat as jailbreak attempts)
        redteam_path = RAW_DATA_DIR / "malicious" / "anthropic_redteam.csv"
        if redteam_path.exists():
            df = pd.read_csv(redteam_path)
            print(f"  ✓ Anthropic Red Team: {len(df):,} rows")
            
            # Extract text
            if 'transcript' in df.columns:
                # Parse conversation to get user prompts
                for transcript in df['transcript'].tolist():
                    if isinstance(transcript, str) and 'Human:' in transcript:
                        # Extract first human message
                        match = re.search(r'Human: (.+?)(?:\n\nAssistant:|$)', transcript, re.DOTALL)
                        if match:
                            text = match.group(1).strip()
                            malicious_prompts.append({
                                'text': text,
                                'label': 2,  # jailbreak
                                'source': 'anthropic_redteam'
                            })
        
        df_malicious = pd.DataFrame(malicious_prompts)
        print(f"\n  📊 Total malicious prompts: {len(df_malicious):,}")
        
        return df_malicious
    
    def load_benign_datasets(self) -> pd.DataFrame:
        """Load and label all benign datasets"""
        
        print("\n🟢 Loading BENIGN datasets...")
        print("-" * 60)
        
        benign_prompts = []
        
        # 1. Alpaca Benign
        alpaca_path = RAW_DATA_DIR / "benign" / "alpaca_benign.csv"
        if alpaca_path.exists():
            df = pd.read_csv(alpaca_path)
            print(f"  ✓ Alpaca Benign: {len(df):,} rows")
            
            # Extract text
            if 'text' in df.columns:
                for text in df['text'].tolist():
                    if isinstance(text, str):
                        benign_prompts.append({
                            'text': text,
                            'label': 0,  # benign
                            'source': 'alpaca'
                        })
        
        # 2. TruthfulQA
        tqa_path = RAW_DATA_DIR / "benign" / "truthful_qa.csv"
        if tqa_path.exists():
            df = pd.read_csv(tqa_path)
            print(f"  ✓ TruthfulQA: {len(df):,} rows")
            
            # Extract text
            if 'text' in df.columns:
                for text in df['text'].tolist():
                    if isinstance(text, str):
                        benign_prompts.append({
                            'text': text,
                            'label': 0,
                            'source': 'truthful_qa'
                        })
        
        df_benign = pd.DataFrame(benign_prompts)
        print(f"\n  📊 Total benign prompts: {len(df_benign):,}")
        
        return df_benign
    
    def combine_and_clean(self, df_malicious: pd.DataFrame, df_benign: pd.DataFrame) -> pd.DataFrame:
        """Combine datasets and perform cleaning"""
        
        print("\n🧹 Cleaning and processing...")
        print("-" * 60)
        
        # Combine datasets
        df_combined = pd.concat([df_malicious, df_benign], ignore_index=True)
        print(f"  ℹ️  Combined dataset: {len(df_combined):,} rows")
        
        # Remove empty rows
        df_combined = df_combined[df_combined['text'].notna()]
        df_combined = df_combined[df_combined['text'].str.len() > 0]
        print(f"  ✓ After removing empty: {len(df_combined):,} rows")
        
        # Clean text
        print("  ⏳ Cleaning text...")
        df_combined['text_cleaned'] = df_combined['text'].apply(self.clean_text)
        
        # Remove rows where cleaned text is empty
        df_combined = df_combined[df_combined['text_cleaned'].str.len() > 10]  # At least 10 chars
        print(f"  ✓ After cleaning: {len(df_combined):,} rows")
        
        # Remove duplicates
        original_count = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['text_cleaned'], keep='first')
        duplicates_removed = original_count - len(df_combined)
        print(f"  ✓ Removed {duplicates_removed:,} duplicates")
        print(f"  ✓ Final dataset: {len(df_combined):,} rows")
        
        # Reorder columns
        df_final = df_combined[['text_cleaned', 'label', 'source', 'text']].copy()
        df_final.columns = ['text', 'label', 'source', 'text_original']
        
        return df_final
    
    def save_processed_data(self, df: pd.DataFrame):
        """Save processed dataset"""
        
        output_path = PROCESSED_DATA_DIR / "prompts.csv"
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Saved processed data to: {output_path}")
        
        # Print label distribution
        print("\n📊 LABEL DISTRIBUTION:")
        print("-" * 60)
        label_counts = df['label'].value_counts().sort_index()
        total = len(df)
        
        for label, count in label_counts.items():
            label_name = {0: 'Benign', 1: 'Injection', 2: 'Jailbreak'}.get(label, 'Unknown')
            percentage = (count / total) * 100
            print(f"  {label} ({label_name:10s}): {count:6,} ({percentage:5.2f}%)")
        
        print(f"\n  Total: {total:,}")
    
    def run(self):
        """Execute complete data cleaning pipeline"""
        
        print("="*60)
        print("DATA CLEANING & LABELING PIPELINE")
        print("="*60)
        
        # Load datasets
        df_malicious = self.load_malicious_datasets()
        df_benign = self.load_benign_datasets()
        
        # Combine and clean
        df_final = self.combine_and_clean(df_malicious, df_benign)
        
        # Save
        self.save_processed_data(df_final)
        
        print("\n" + "="*60)
        print("✅ PHASE 1 COMPLETE!")
        print("="*60)
        print("\nNext step: PHASE 2 - Feature Engineering")


def main():
    """Main execution"""
    cleaner = DataCleaner()
    cleaner.run()


if __name__ == "__main__":
    main()
