# AI Prompt Security Detection System

A machine learning system to detect **Prompt Injection** and **Jailbreak** attacks in Large Language Models (LLMs).

## 🎯 Project Goal

Build a preamble security classifier that acts as a guardrail before an LLM.
- **Input**: User prompt
- **Output**: `ALLOW` (Benign) or `BLOCK` (Malicious)

**Labels:**
- `0` = Benign
- `1` = Prompt Injection
- `2` = Jailbreak

## 🚀 How to Run (Fresh Install)

Since dataset files and models are large, they are not stored in the repository. You must regenerate them using the provided scripts.

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/Harshith1004/ai_prompt_security.git
cd ai_prompt_security

# Run setup script (Mac/Linux)
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Build Pipeline (Regenerate Data & Models)
Run these commands in order to create the models from scratch:

```bash
# 1. Download Datasets (Alpaca, TruthfulQA, Jailbreaks)
python src/download_datasets.py

# 2. Clean and Label Data
python src/data_cleaning.py

# 3. Extract Features (Lexical + S-BERT Embeddings)
# ⚠️ This takes ~2-5 minutes
python src/feature_extraction.py

# 4. Train Models (Logistic Regression, Random Forest, MLP)
python src/train_models.py
```

### 3. Run the Guardrail Demo
Once models are trained, you can use the interactive scanner:

```bash
python src/guardrail.py
```

## 📂 Project Structure

```
ai_prompt_security/
├── data/             # (Generated locally)
├── models/           # (Generated locally)
├── notebooks/        # Experiments
├── results/          # Evaluation metrics & plots
├── src/              # Source code
│   ├── download_datasets.py   # Step 1
│   ├── data_cleaning.py       # Step 2
│   ├── feature_extraction.py  # Step 3
│   ├── train_models.py        # Step 4
│   ├── evaluate_models.py     # Step 5
│   └── guardrail.py           # Demo App
└── requirements.txt
```

## 📊 Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **DistilBERT Head (MLP)** | **97.9%** | **0.99** |
| Logistic Regression | 96.2% | 0.98 |
| Random Forest | 96.0% | 0.98 |

## 📁 Source Datasets

**Malicious:**
- `deepset/prompt-injections`
- `Anthropic/hh-rlhf` (Red Team)
- `rubend18/ChatGPT-Jailbreak-Prompts`

**Benign:**
- `tatsu-lab/alpaca`
- `truthful_qa`

## 📝 License
Research project for educational purposes.
