# Phase 1: Data Collection & Cleaning - Report

**Date:** 2026-06-XX
**Status:** ✅ Complete

## 📊 Final Dataset Statistics

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| Benign | 0 | 52,810 | 58.36% |
| Injection | 1 | 544 | 0.60% |
| Jailbreak | 2 | 37,133 | 41.04% |
| **Total** | | **90,487** | **100%** |

## 📁 Source Files

### Malicious Data
- `data/raw/malicious/prompt_injections_deepset.csv` (Deepset)
- `data/raw/malicious/anthropic_redteam.csv` (Anthropic HH-RLHF)
- `data/raw/malicious/jailbreak_prompts.csv` (ChatGPT Jailbreaks)

### Benign Data
- `data/raw/benign/alpaca_benign.csv` (Stanford Alpaca)
- `data/raw/benign/truthful_qa.csv` (TruthfulQA)

## 🧹 Processing Steps Applied
- Lowercased all text
- Removed URLs
- Removed HTML tags
- Removed emojis
- Removed duplicates (1,602 duplicates removed)
- Filtered short text (< 10 chars)

## 📍 Output
- **File:** `data/processed/prompts.csv`
- **Columns:** `text`, `label`, `source`, `text_original`
