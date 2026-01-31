# PHASE 1: DATA COLLECTION & CLEANING

## ✅ Deliverables Checklist

### 1. Project Structure ✅
- [x] Created `ai_prompt_security/` folder
- [x] Created `data/raw/` directory
- [x] Created `data/processed/` directory
- [x] Created `notebooks/` directory
- [x] Created `src/` directory
- [x] Created `models/` directory
- [x] Created `results/` directory

### 2. Dataset Collection
#### Malicious Datasets (Injection + Jailbreak)
- [ ] Prompt Injection dataset (Liu et al. / deepset)
- [ ] JailbreakBench dataset
- [ ] Anthropic red team prompts
- [ ] OpenAI red team examples
- [ ] Awesome-Jailbreak-Prompts (GitHub)

#### Benign Datasets
- [ ] ShareGPT dataset
- [ ] OpenAI eval benign prompts
- [ ] Kaggle chat datasets / LMSYS

### 3. Data Labeling
- [ ] Label malicious prompts with:
  - `1` = prompt_injection
  - `2` = jailbreak
- [ ] Label benign prompts with:
  - `0` = benign

### 4. Data Cleaning
- [ ] Convert all text to lowercase
- [ ] Remove duplicate prompts
- [ ] Remove URLs from text
- [ ] Remove emojis from text
- [ ] Remove HTML tags from text
- [ ] Remove empty rows (< 10 characters)
- [ ] Strip extra whitespace

### 5. Output
- [ ] Save final dataset to `data/processed/prompts.csv`
- [ ] Verify CSV has columns: `text`, `label`, `source`
- [ ] Verify label distribution is reasonable

## 📊 Expected Dataset Statistics

**Minimum targets:**
- Benign prompts: 5,000+
- Injection prompts: 1,000+
- Jailbreak prompts: 500+
- Total: 6,500+ samples

## 🚀 Execution Steps

### Step 1: Environment Setup
```bash
cd ai_prompt_security
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Step 2: Download Datasets
```bash
python src/download_datasets.py
```

**Expected output:**
- Raw CSV files in `data/raw/malicious/`
- Raw CSV files in `data/raw/benign/`

### Step 3: Clean and Label Data
```bash
python src/data_cleaning.py
```

**Expected output:**
- Single file: `data/processed/prompts.csv`
- Console output showing label distribution

### Step 4: Verify Results
```bash
# Check file exists
ls -lh data/processed/prompts.csv

# Quick preview
head -20 data/processed/prompts.csv
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/01_data_collection.ipynb
```

## 🔍 Quality Checks

Before moving to Phase 2, verify:

1. **File exists**: `data/processed/prompts.csv`
2. **No duplicates**: Run duplicate check
3. **Balanced classes**: Check label distribution (shouldn't be 99% one class)
4. **Clean text**: Sample 10 random rows, verify no HTML/URLs/emojis
5. **Reasonable length**: Most prompts should be 10-500 characters

## ⚠️ Common Issues

### Issue: Dataset download fails
**Solution**: Check internet connection, try manual download from HuggingFace

### Issue: Missing columns in CSV
**Solution**: Check the column names in raw datasets, update `data_cleaning.py` column references

### Issue: Class imbalance (too many benign)
**Solution**: Downsample benign class or upsample malicious classes

## 📝 Documentation

Document in your dissertation:
- Total samples collected per dataset
- Data sources (with citations)
- Cleaning steps applied
- Final label distribution
- Any manual labeling performed

## ✅ Phase 1 Completion Criteria

Phase 1 is complete when:
- [ ] `prompts.csv` exists in `data/processed/`
- [ ] Contains at least 6,500 samples
- [ ] All three labels (0, 1, 2) are present
- [ ] No duplicate prompts
- [ ] Text is cleaned (no URLs, HTML, emojis)
- [ ] You can load and view the data successfully

---

## 🎯 Ready for Phase 2?

Once all checkboxes above are complete, you're ready to proceed to:
**PHASE 2: FEATURE ENGINEERING**

Do NOT proceed until Phase 1 is fully complete.
