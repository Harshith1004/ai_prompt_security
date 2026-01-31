# PHASE 2: FEATURE ENGINEERING - PLAN

 **Goal:** Convert raw text prompts into numerical vectors for machine learning models.

## 🛠 Feature Sets to Extract

### 1. Lexical Features (Statistical)
Simple but effective features based on text statistics.
- [ ] **Length**: Character count, word count
- [ ] **Capitalization**: Ratio of uppercase letters (shouting/imperatives)
- [ ] **Punctuation**: Count of `!`, `?`, `"`, `{`, `}` (often used in code injections)
- [ ] **Special Characters**: Count of `@`, `#`, `$`, `%`
- [ ] **Keyword Presence**: Binary flags for suspicious words:
  - `ignore`, `override`, `system`, `developer`, `model`, `prior`
  - `jailbreak`, `mode`, `unfiltered`, `dan`

### 2. Structural Features (Pattern-based)
Detecting the *structure* of attacks.
- [ ] **Role Play Indicators**: Matches for `You are`, `Act as`, `Pretend`
- [ ] **Chat Template Injection**: Matches for `User:`, `Assistant:`, `System:`
- [ ] **Code Injection**: Presence of code blocks or programming syntax (`def `, `import `, `print(`)

### 3. Semantic Features (Embeddings)
Capturing the *meaning* of the text.
- [ ] **Sentence-BERT Embeddings**: Use `all-MiniLM-L6-v2` (fast & effective)
  - Output: 384-dimensional vector for each prompt

## 📝 Execution Plan

1. **Create `src/feature_extraction.py`**:
   - Class `FeatureExtractor`
   - Setup `scikit-learn` transformers and pipelines
   - Setup `sentence-transformers` model

2. **Process Data**:
   - Load `data/processed/prompts.csv`
   - Apply extraction in batches (embeddings can be slow)
   - Combine all features into a single feature matrix ($X$)

3. **Save Features**:
   - Save $X$ (features) and $y$ (labels) as `.npz` (NumPy arrays) or `.parquet`
   - `data/processed/X_features.npz`
   - `data/processed/y_labels.npy`

## ⏱ Estimated Time
- **Lexical/Structural**: < 1 minute
- **Embeddings**: ~15-30 minutes (on CPU) for 90k samples

## 🚀 Next Command
```bash
python src/feature_extraction.py
```
