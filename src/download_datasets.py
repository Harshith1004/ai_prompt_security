"""
Automated Dataset Downloader for Prompt Security Project

This script downloads all required datasets using HuggingFace datasets library.
"""

import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_prompt_injection_dataset():
    """Download deepset/prompt-injections from HuggingFace"""
    print("\n📥 Downloading: Prompt Injection Dataset (deepset)")
    print("-" * 60)
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("deepset/prompt-injections")
        
        # Convert to pandas DataFrame
        df_train = pd.DataFrame(dataset['train'])
        
        # Save to CSV
        output_path = RAW_DATA_DIR / "malicious" / "prompt_injections_deepset.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(output_path, index=False)
        
        print(f"✅ Saved: {output_path}")
        print(f"   Rows: {len(df_train)}")
        print(f"   Columns: {list(df_train.columns)}")
        
        return df_train
        
    except Exception as e:
        print(f"❌ Error downloading prompt injections: {e}")
        return None


def download_anthropic_redteam():
    """Download Anthropic HH-RLHF red team dataset"""
    print("\n📥 Downloading: Anthropic Red Team Dataset")
    print("-" * 60)
    
    try:
        # Load the red-team-attempts split
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
        
        # Convert to pandas
        df = pd.DataFrame(dataset['train'])
        
        # Save to CSV
        output_path = RAW_DATA_DIR / "malicious" / "anthropic_redteam.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✅ Saved: {output_path}")
        print(f"   Rows: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading Anthropic dataset: {e}")
        return None


def download_alpaca_benign():
    """Download Alpaca dataset for benign prompts"""
    print("\n📥 Downloading: Alpaca (Benign Prompts)")
    print("-" * 60)
    
    try:
        # Load Alpaca dataset
        dataset = load_dataset("tatsu-lab/alpaca")
        
        # Convert to pandas
        df = pd.DataFrame(dataset['train'])
        
        # Rename instruction to text
        df['text'] = df['instruction']
        
        # Save to CSV
        output_path = RAW_DATA_DIR / "benign" / "alpaca_benign.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✅ Saved: {output_path}")
        print(f"   Rows: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading Alpaca: {e}")
        return None


def download_truthful_qa_benign():
    """Download TruthfulQA for benign prompts"""
    print("\n📥 Downloading: TruthfulQA (Benign Prompts)")
    print("-" * 60)
    
    try:
        # Load TruthfulQA
        dataset = load_dataset("truthful_qa", "generation")
        
        df = pd.DataFrame(dataset['validation'])
        
        # Rename question to text
        df['text'] = df['question']
        
        # Save to CSV
        output_path = RAW_DATA_DIR / "benign" / "truthful_qa.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✅ Saved: {output_path}")
        print(f"   Rows: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading TruthfulQA: {e}")
        return None


def download_jailbreak_prompts():
    """Download jailbreak prompts from available sources"""
    print("\n📥 Downloading: Jailbreak Prompts")
    print("-" * 60)
    
    try:
        # Try to load from rubend18/ChatGPT-Jailbreak-Prompts
        dataset = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts")
        
        df = pd.DataFrame(dataset['train'])
        
        # Save to CSV
        output_path = RAW_DATA_DIR / "malicious" / "jailbreak_prompts.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✅ Saved: {output_path}")
        print(f"   Rows: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading jailbreak prompts: {e}")
        return None


def create_dataset_summary():
    """Create a summary of downloaded datasets"""
    print("\n" + "="*60)
    print("DATASET DOWNLOAD SUMMARY")
    print("="*60 + "\n")
    
    malicious_dir = RAW_DATA_DIR / "malicious"
    benign_dir = RAW_DATA_DIR / "benign"
    
    print("🔴 MALICIOUS DATASETS:")
    if malicious_dir.exists():
        for file in malicious_dir.glob("*.csv"):
            df = pd.read_csv(file)
            print(f"  ✓ {file.name}: {len(df):,} rows")
    
    print("\n🟢 BENIGN DATASETS:")
    if benign_dir.exists():
        for file in benign_dir.glob("*.csv"):
            df = pd.read_csv(file)
            print(f"  ✓ {file.name}: {len(df):,} rows")
    
    print("\n" + "="*60)


def main():
    """Main download execution"""
    print("="*60)
    print("AUTOMATED DATASET DOWNLOADER")
    print("="*60)
    print("\n⚠️  Note: This may take several minutes...")
    print("⚠️  Ensure you have internet connection\n")
    
    # Download malicious datasets
    download_prompt_injection_dataset()
    download_anthropic_redteam()
    download_jailbreak_prompts()
    
    # Download benign datasets
    download_alpaca_benign()
    download_truthful_qa_benign()
     
    # Jailbreak prompts (keep this one)
    download_jailbreak_prompts()
    
    # Create summary
    create_dataset_summary()
    
    print("\n✅ Dataset download complete!")
    print(f"📁 All files saved to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
