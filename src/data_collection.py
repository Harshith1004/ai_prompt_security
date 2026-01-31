"""
Data Collection Module for Prompt Injection & Jailbreak Detection

This script handles downloading and organizing datasets from multiple sources.

Dataset Sources:
1. MALICIOUS PROMPTS:
   - Prompt Injection dataset (Liu et al.)
   - JailbreakBench
   - Anthropic red team prompts
   - OpenAI red team examples
   - Awesome-Jailbreak-Prompts (GitHub)

2. BENIGN PROMPTS:
   - ShareGPT
   - OpenAI eval benign
   - Kaggle chat datasets
"""

import os
import json
import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class DatasetCollector:
    """Handles downloading and initial organization of datasets"""
    
    def __init__(self):
        self.datasets = {
            'malicious': [],
            'benign': []
        }
    
    def download_from_url(self, url: str, filename: str, category: str) -> bool:
        """
        Download a dataset from a URL
        
        Args:
            url: Dataset URL
            filename: Local filename to save as
            category: 'malicious' or 'benign'
        """
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filepath = RAW_DATA_DIR / category / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {str(e)}")
            return False
    
    def download_github_dataset(self, repo_url: str, target_files: List[str], category: str):
        """Download specific files from a GitHub repository"""
        # This would download from GitHub repos like Awesome-Jailbreak-Prompts
        print(f"📦 GitHub repo: {repo_url}")
        print("Note: You may need to clone this repository manually")
        pass
    
    def collect_all_datasets(self):
        """Main method to collect all required datasets"""
        
        print("\n" + "="*60)
        print("PHASE 1: DATA COLLECTION")
        print("="*60 + "\n")
        
        print("📋 REQUIRED DATASETS:")
        print("\n🔴 MALICIOUS PROMPTS:")
        print("  1. Prompt Injection dataset (Liu et al.)")
        print("     → HuggingFace: deepset/prompt-injections")
        print("  2. JailbreakBench")
        print("     → GitHub: JailbreakBench/JailbreakBench")
        print("  3. Anthropic red team prompts")
        print("     → HuggingFace: Anthropic/hh-rlhf (red-team subset)")
        print("  4. OpenAI red team examples")
        print("     → OpenAI Evals GitHub repo")
        print("  5. Awesome-Jailbreak-Prompts")
        print("     → GitHub: prompt-engineering/awesome-jailbreak-prompts")
        
        print("\n🟢 BENIGN PROMPTS:")
        print("  1. ShareGPT")
        print("     → HuggingFace: anon8231489123/ShareGPT_Vicuna_unfiltered")
        print("  2. OpenAI eval benign")
        print("     → OpenAI Evals GitHub repo")
        print("  3. Kaggle chat datasets")
        print("     → Kaggle: Chat conversations datasets")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("\nWe'll download these datasets using:")
        print("  • HuggingFace datasets library (for HF datasets)")
        print("  • Git clone (for GitHub repos)")
        print("  • Direct download (for public files)")
        
        return self.datasets


def main():
    """Main execution function"""
    collector = DatasetCollector()
    collector.collect_all_datasets()
    
    print("\n✅ Data collection module initialized")
    print(f"📁 Raw data directory: {RAW_DATA_DIR}")
    print(f"📁 Processed data directory: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
