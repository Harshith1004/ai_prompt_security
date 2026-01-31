# Dissertation Structure & Outline

**Title:** Automated Detection of Prompt Injection and Jailbreak Attacks in Large Language Models using Hybrid Feature Extraction

## Chapter 1: Threat Model & Motivation
- **The Rise of LLMs:** Brief adoption history.
- **The Security Gap:** Why traditional WAFs fail on natural language.
- **Attack Taxonomy:**
  - *Direct Injection:* Overriding system instructions.
  - *Jailbreaking:* Bypassing safety filters (DAN, roleplay).
  - *Obfuscation:* Base64, translation, payload splitting.
- **Research Question:** Can we build a preamble detector that generalizes to unseen attacks without degrading user experience?

## Chapter 2: Literature Review
- **Adversarial ML:** Gradient-based attacks (GCG).
- **Existing Defenses:** Perplexity checks, retraining, separate guard models (Llama Guard).
- **Gaps:** Latency issues with Llama Guard; lack of robustness in perplexity filters.

## Chapter 3: Dataset & Methodology
- **Data Collection:**
  - Malicious: Anthropic Red Team, JailbreakBench, Deepset Injections.
  - Benign: Alpaca, TruthfulQA.
- **Preprocessing:** Cleaning pipeline (deduplication, normalization).
- **Feature Engineering (The Hybrid Approach):**
  - *Lexical:* Statistical anomalies (keywords, caps).
  - *Structural:* Pattern recognition (stacking, nesting).
  - *Semantic:* Dense embeddings (Sentence-BERT).
- **Model Selection:** Comparison of LR, RF, and MLP (DistilBERT head).

## Chapter 4: Results
- **Performance Metrics:**
  - Accuracy: 97.9% (DistilBERT Head).
  - ROC-AUC: 0.99+.
  - False Positive Rate: ~1.8% (Critical for UX).
- **Comparison:** Why MLP outperformed RF (Semantic understanding vs feature rules).
- **Confusion Matrix Analysis:** Where does it fail? (Subtle jailbreaks).

## Chapter 5: Stress Testing
- **Paraphrasing:** Testing with synonymous attacks.
- **Multilingual:** Does it work on Spanish/French attacks? (Limitations of S-BERT).
- **Adaptive Attacks:** Can we bypass our own detector?

## Chapter 6: Ethics & Limitations
- **Dual Use:** Can this research help attackers?
- **Bias:** Does the benign filter block legitimate contentious topics?
- **Performance:** Latency overhead of S-BERT (approx 20ms).

## Chapter 7: Conclusion
- Summary of contributions.
- Future work: End-to-end differentiable defenses.
