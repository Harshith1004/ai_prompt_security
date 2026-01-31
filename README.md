# AI Prompt Security Detection System

Machine learning system to detect prompt injection and jailbreak attacks in large language models.

## 🎯 Project Goal

Build a classifier that labels user prompts as:
- `0` = benign
- `1` = prompt_injection  
- `2` = jailbreak

## 📁 Project Structure

```
ai_prompt_security/
├── data/
│   ├── raw/          # Original downloaded datasets
│   └── processed/    # Cleaned and labeled data
├── notebooks/        # Jupyter notebooks for experimentation
├── src/             # Source code
├── models/          # Trained models
├── results/         # Evaluation results and reports
└── requirements.txt
```

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets (Phase 1)

```bash
python src/download_datasets.py
```

This will download:
- **Malicious prompts**: Prompt injections, jailbreaks, red team attacks
- **Benign prompts**: ShareGPT, LMSYS chat data

### 3. Clean & Process Data

```bash
python src/data_cleaning.py
```

### 4. Train Models (Coming in Phase 3)

```bash
python src/train_model.py
```

## 📊 Dataset Sources

### Malicious Prompts
- deepset/prompt-injections
- Anthropic/hh-rlhf (red team)
- rubend18/ChatGPT-Jailbreak-Prompts

### Benign Prompts
- anon8231489123/ShareGPT_Vicuna_unfiltered
- lmsys/lmsys-chat-1m

## 🔬 Project Phases

- [x] **Phase 1**: Data Collection & Cleaning
- [ ] **Phase 2**: Feature Engineering
- [ ] **Phase 3**: Model Training
- [ ] **Phase 4**: Evaluation
- [ ] **Phase 5**: Error Analysis
- [ ] **Phase 6**: Guardrail Demo
- [ ] **Phase 7**: Dissertation

## 📝 License

Research project for educational purposes.
