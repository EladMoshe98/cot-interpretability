# Do Language Models Actually Reason?
## Mechanistic Interpretability of Chain-of-Thought Reasoning in LLMs

**MSc Thesis — IE School of Science and Technology, Madrid, 2026**

---

### Research Question

When a language model is asked to "show its work" before giving an answer (chain-of-thought / CoT prompting), does it genuinely compute the answer through the written reasoning steps — or does it already know the answer and produce post-hoc justifications?

This project investigates this question by probing the **internal representations** of [Gemma-2-2B-IT](https://huggingface.co/google/gemma-2-2b-it) across 12 experiments on 400 GSM8K elementary math problems.

---

### Key Findings

| Finding | Evidence |
|---|---|
| **CoT is a mode switch, not just more text.** Writing reasoning triggers a qualitatively distinct computational regime, detectable *before* any reasoning word is generated. | Exp 4–5: AUROC = 1.000 at every layer/position |
| **Correctness is decodable from internal state.** A simple linear classifier on residual-stream activations predicts answer correctness at up to 80% AUROC. | Exp 1: AUROC = 0.801 at Layer 18 |
| **The signal is diffuse and uninterpretable.** It lives in the SAE error term — not in any named Gemma Scope feature. NLA text descriptions lose it entirely. | Exp 2: −0.09 AUROC via features; Exp 9–10: null result |
| **Correctness is distributed across the whole chain.** Computation tokens and reasoning connectors carry identical signal; full-chain averaging outperforms any single position. | Exp 11: AUROC = 0.750 for each token type |
| **Failure mode: computing right, writing wrong.** In 78% of incorrect chains, the correct digit reaches high probability mid-chain (max P = 0.83) before being overwritten. | Exp 12: Late-stage carry failure |

---

### Experiments

| Exp | Name | n | Key Result |
|---|---|---|---|
| 1 | Residual Stream Regression | 400 | AUROC = 0.801 (CoT, L18, Pos A); Δ = +0.119 vs. NoCoT |
| 2 | SAE Feature Decomposition | 400 | Signal in error term (AUROC 0.770); features lose 0.09 AUROC |
| 3 | Log-Probability Baseline | 400 | AUROC = 0.692 (well below mechanistic signal) |
| 4 | Flipped-Problems Classifier | 186 | AUROC = 1.000 at all layers & positions |
| 5 | Mode-Switch Control | 236 | AUROC = 1.000 (format separable on correct-only problems) |
| 6 | Shuffled-Label Baseline | 400 | Real AUROCs 0.17–0.27 above shuffled baselines |
| 7 | Gemma-3-27B Accuracy Reference | 400 | 91.5% CoT vs. 31.5% NoCoT (+60 pp) |
| 8 | NLA Qualitative Examples | 10 | Correct chains name specific domain; incorrect chains misattribute |
| 9 | NLA Embedding Probe | 58 | CV AUROC = 0.363 (below chance; DEMONSTRATIVE) |
| 10 | LLM Judge from NLA | 58 | 48.3% accuracy, TP = 0 (DEMONSTRATIVE) |
| 11 | Token-Type Attribution | 80 | Computation = reasoning connectors = 0.750 AUROC |
| 12 | Answer Probability Trajectory | 68 | 78% of wrong chains peak on correct digit mid-reasoning |

---

### Models & Data

| Item | Details |
|---|---|
| Primary model | `google/gemma-2-2b-it` |
| Reference model | `google/gemma-3-27b` (Exp 7–10 via Neuronpedia NLA API) |
| Dataset | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) test set, n = 400, seed = 42 |
| Interpretability tool | [Gemma Scope](https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-interpret-gemma-2/) SAE, Layer 12, 16 384 features |
| NLA API | [Neuronpedia](https://www.neuronpedia.org/) Natural Language Autoencoder |
| Classifier | L2-regularised logistic regression, C = 0.1, 5-fold stratified CV, AUROC metric |

---

### Repository Structure

```
cot-interpretability/
├── Experimental_Results_Summary.docx   ← full write-up with all figures (professor-readable)
└── experiments/
    ├── exp1_residual_stream_regression/
    │   ├── exp1_residual_stream_regression.ipynb
    │   ├── figures/          ← PNG outputs
    │   └── cache/            ← intermediate data (parquet, json, npy)
    ├── exp2_sae_decomposition/
    ├── exp3_logprob_mean/
    ├── exp4_flipped_problems_classifier/
    ├── exp5_mode_switch_control/
    ├── exp6_shuffled_label_baseline/
    ├── exp7_gemma3_reference_with_NLA/   ← covers Exp 7, 8, 9, 10
    ├── exp11_token_type_attribution/
    └── exp12_answer_probability_trajectory/
        ├── e12_examples_interactive.html  ← interactive Plotly visualization
        └── e12_single_digit_problems.csv
```

> **Note:** Experiments 8, 9, and 10 are contained within the `exp7_gemma3_reference_with_NLA` notebook as they share the same model (Gemma-3-27B) and data pipeline.

---

### How to Run

All notebooks run on Google Colab with a T4 GPU (free tier) or equivalent.

1. **Install dependencies** — the first cell in each notebook installs required packages (`transformers`, `accelerate`, `scipy`, `plotly`, etc.)
2. **Mount Google Drive** — notebooks cache model outputs to Drive so you don't re-run the forward passes
3. **Run cells top to bottom** — each notebook is self-contained

> ⚠️ Experiments 1–6 and 11–12 load `google/gemma-2-2b-it` from HuggingFace and require ~5 GB GPU memory. Experiments 7–10 use the Neuronpedia NLA API (API key required).

---

### Results Summary Document

A formal write-up with all figures, statistical annotations, analysis, limitations, and further work is available as [`Experimental_Results_Summary.docx`](./Experimental_Results_Summary.docx).

It includes:
- Background on LLMs, CoT, mechanistic interpretability, SAEs, and NLAs (accessible to non-specialists)
- All 10 results tables with p-values and AUROC values
- 15 embedded figures
- Cross-experiment analysis and big-picture LLM implications
- Research limitations (9 items)
- Further work (7 items)

---

### Citation

```bibtex
@misc{moshe2026cot,
  title  = {Mechanistic Interpretability of Chain-of-Thought Reasoning in Language Models},
  author = {Elad Moshe},
  year   = {2026},
  school = {IE School of Science and Technology},
  note   = {MSc Research Capstone}
}
```

---

### License

MIT — see [LICENSE](./LICENSE)
