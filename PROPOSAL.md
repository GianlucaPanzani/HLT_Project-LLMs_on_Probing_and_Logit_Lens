# Project Specification: Inside a Language Model — Probing and the Logit Lens (Group 11)

> **Instructions for the AI Coding Assistant:**  
> You are acting as a Lead Machine Learning Engineer and NLP Research Architect. Based on the following complete technical specification extracted from our official project presentation, generate a **comprehensive, step-by-step programming TODO list**, a **modular Python repository structure**, and **concrete implementation guidelines** divided by workstream and project phase.

---

## 1. Project Overview & Team Allocation

* **Course:** Human Language Technologies (HLT) — Group 11
* **Team Members (5):** Matteo Salvini, Marcin Czarnecki, Gianluca Panzani, Arcangelo Chinè, Filippo Gigante.
* **Core Objective:** Investigate how morphosyntactic information (Universal Part-of-Speech tags, UPOS) and lexical predictions evolve layer-by-layer across frozen Transformer models using **Supervised Probing** and the **Mechanistic Lens Spectrum** (Standard Logit Lens, Tuned Lens, Jacobian Lens).
* **Work Schedule & Parallel Split (Phase 1: Nov 13 – Nov 18):**
  * **Members 1 & 2:** General Probing Analysis, Baselines (Layer 0, Random Weights, Control Tasks), Linear vs. MLP Probes, and Architecture Comparison (Causal vs. Bidirectional).
  * **Member 3:** Model Compression & Linguistic Hierarchy (`BERT-base` vs. `DistilBERT`, 1:2 Layer Mapping, and Centered Kernel Alignment - CKA).
  * **Member 4:** Cross-Linguistic Analysis (English vs. Italian across Single Multilingual Models and Paired Monolingual Models, including `mGPT` depth/dimensionality normalization).
  * **Member 5:** Decoder Lens Spectrum (`GPT-2` and `GePpeTto` with Standard Logit Lens, Tuned Lens, and Jacobian Lens) & Cross-Referencing with Probing curves.
  * **All Members (Phase 2–Final: Nov 19 – Dec 3):** Pipeline integration, cross-analysis plots, and an interactive **Streamlit Demo**.

---

## 2. Models Inventory (Strictly Frozen Backbones: $\nabla_{\theta_f} = 0$)

All pretrained language models must be loaded via Hugging Face `transformers` with `output_hidden_states=True` and all parameters strictly frozen (`param.requires_grad = False`).

| Role / Analysis | Model Name (Hugging Face ID) | Architecture | Layers ($L$) | Hidden Dim ($d$) | Language |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Bidirectional Baseline & Teacher** | `bert-base-cased` | Encoder (MLM) | $12$ (+ Layer 0) | $768$ | English |
| **Compressed Student** | `distilbert-base-cased` | Distilled Encoder | $6$ (+ Layer 0) | $768$ | English |
| **Causal Decoder (EN)** | `gpt2` | Decoder (CLM) | $12$ (+ Layer 0) | $768$ | English |
| **Monolingual Encoder (IT)** | `dbmdz/bert-base-italian-cased` | Encoder (MLM) | $12$ (+ Layer 0) | $768$ | Italian |
| **Monolingual Decoder (IT)** | `LorenzoDeMattei/GePpeTto` | Decoder (CLM) | $12$ (+ Layer 0) | $768$ | Italian |
| **Multilingual Encoder** | `bert-base-multilingual-cased` | Encoder (MLM) | $12$ (+ Layer 0) | $768$ | Multilingual (EN/IT) |
| **Multilingual Decoder** | `ai-forever/mGPT` | Scaled Decoder (CLM) | $24$ (+ Layer 0) | $2048$ | Multilingual (61 langs) |

---

## 3. Datasets & Preprocessing Pipeline

### 3.1 Probing Datasets (Supervised Morphosyntactic Extraction)
* **Source:** Universal Dependencies (UD).
* **Treebanks:**
  * **English:** `UD_English-EWT` (`train`, `dev`, `test` splits).
  * **Italian:** `UD_Italian-ISDT` (`train`, `dev`, `test` splits).
* **Extracted Fields:** Token strings, sentence boundaries, and gold `UPOS` tags (17 universal classes).
* **Subword Alignment Rule:** Pretrained tokenizers (WordPiece / BPE) split words into multiple subwords. Extract the hidden state $h_i^{(l)}$ corresponding to the **first subword token** of each gold UD word (or compare with mean-pooling across subwords of the same word), discarding special tokens (`[CLS]`, `[SEP]`, `<|endoftext|>`).

### 3.2 Logit Lens Datasets (Next-Token & Entropy Trajectories)
* **Parallel & Syntactic Continuous Text (Test):**
  * `UD-PUD` parallel splits (`UD_English-PUD` and `UD_Italian-PUD`) to evaluate both probing milestones and lens entropy trajectories on aligned EN-IT sentences without pretraining contamination.
  * `WikiText-103` (test split) for English natural continuous text evaluation.
* **Tuned Lens Calibration (Self-Supervised Training):**
  * **For `gpt2`:** Load official pretrained Tuned Lens checkpoints (via `tuned-lens` library) or calibrate on `WikiText-103` train split.
  * **For `GePpeTto`:** Train lightweight layer-wise affine translators using self-supervised KL divergence minimization on `UD_Italian-ISDT` (`train` split), ensuring zero overlap with the test splits.

---

## 4. Detailed Workstreams & Technical Requirements

### Workstream 1: Core Probing Pipeline, Baselines & Architecture Comparison (2 People)
1. **Feature Extraction Engine:**
   * Pass input sequences $x = [t_1, \dots, t_n]$ through frozen backbone $f$.
   * Extract token-level activation vectors $h_i^{(l)} \in \mathbb{R}^d$ for every layer $l \in \{0, 1, \dots, L\}$.
2. **Three Mandatory Probing Baselines:**
   * **Baseline 1 — Layer 0 Lookup:** Probe the non-contextual static embedding layer ($l = 0$) to measure lexical lookup performance prior to self-attention.
   * **Baseline 2 — Random Weights:** Instantiate the model configuration (`AutoModel.from_config(config)`) with untrained random weights and run the full probing pipeline across all layers to isolate purely architectural inductive bias.
   * **Baseline 3 — Control Tasks (Hewitt & Liang, 2019):** Construct a random deterministic mapping $C: V_{\text{word}} \to \{1, \dots, 17\}$ assigning each unique surface word type a random UPOS label (sampled according to the empirical UPOS marginal distribution). Train identical probes on these control labels to compute $\text{Acc}_{\text{control}}^{(l)}$.
3. **Probe Architectures Evaluated:**
   * **Linear Probe:** $L_2$-regularized multinomial logistic regression classifier:
     $$\hat{z} = \text{Softmax}\!\left(W_{\text{probe}} h_i^{(l)} + b_{\text{probe}}\right)$$
   * **MLP Extension:** Single hidden-layer Multi-Layer Perceptron with ReLU activation tested strictly under Selectivity control to verify whether non-linear encoding increases selectivity or merely inflates memorization.
4. **Causal vs. Bidirectional Comparison (`gpt2` vs. `bert-base-cased`):**
   * Compare how UPOS information emerges across layers when attending only to preceding context (`gpt2`) versus bidirectional left+right context (`bert-base-cased`).
   * Evaluate both current-token UPOS ($y_t$) and next-token UPOS ($y_{t+1}$) decodability to prepare the bridge with the Logit Lens.

---

### Workstream 2: Model Compression & Linguistic Hierarchy (1 Person)
1. **Models Compared:** `bert-base-cased` ($L=12$) vs. `distilbert-base-cased` ($L=6$).
2. **Rigid 1:2 Layer Mapping:**
   * Compare DistilBERT layers $\{0, 1, 2, 3, 4, 5, 6\}$ directly against BERT even layers $\{0, 2, 4, 6, 8, 10, 12\}$ in terms of UPOS Accuracy, Selectivity, and Center of Gravity ($\text{CoG}_{\text{norm}}$).
3. **Empirical Geometric Validation via Centered Kernel Alignment (CKA):**
   * Because knowledge distillation transfers the output function rather than intermediate geometry, test whether DistilBERT Layer $l$ actually corresponds to BERT Layer $2l$.
   * Extract representation matrices $X \in \mathbb{R}^{N \times d}$ (BERT layer $i$) and $Y \in \mathbb{R}^{N \times d}$ (DistilBERT layer $j$) over $N$ aligned tokens from the test set.
   * Compute the $13 \times 7$ **Linear CKA similarity matrix** using mean-centered activations $\tilde{X} = H X$ and $\tilde{Y} = H Y$:
     $$\text{Linear CKA}(X, Y) = \frac{\|\tilde{Y}^T \tilde{X}\|_F^2}{\|\tilde{X}^T \tilde{X}\|_F \, \|\tilde{Y}^T \tilde{Y}\|_F}$$
   * Plot the CKA heatmap and identify the true optimal geometric mapping across layers.

---

### Workstream 3: Cross-Linguistic Analysis — English vs. Italian (1 Person)
1. **Research Question:** Does richer morphological inflection in Italian vs. rigid word order in English shift the relative layer depth where morphosyntactic features peak?
2. **Paradigm A — Single Multilingual Model (`bert-base-multilingual-cased` & `ai-forever/mGPT`):**
   * Probe the exact same frozen network on English (`UD_English-EWT` / `UD_English-PUD`) and Italian (`UD_Italian-ISDT` / `UD_Italian-PUD`).
   * **Handling `mGPT` Scale Discrepancy:** Because `mGPT` has $L=24$ layers and hidden size $d=2048$ (vs. $L=12, d=768$ in `gpt2` and `GePpeTto`), raw probing accuracy curves cannot be compared directly due to Cover's theorem (higher dimensionality inflates linear separability).
   * Apply the three mandatory normalization metrics: **Relative Layer Depth ($\tau$)**, **Normalized Center of Gravity ($\text{CoG}_{\text{norm}}$)**, and **Probing Selectivity ($S^{(l)}$)**.
   * *Memory Optimization Note:* Run `mGPT` (1.3B parameters) in `torch.float16` or `torch.bfloat16` during feature extraction.
3. **Paradigm B — Paired Monolingual Models:**
   * Encoders: `bert-base-cased` (EN) vs. `dbmdz/bert-base-italian-cased` (IT).
   * Decoders: `gpt2` (EN) vs. `LorenzoDeMattei/GePpeTto` (IT).
   * Compare relative abstraction trajectories and compute cross-lingual Center of Gravity shifts ($\Delta \text{CoG} = \text{CoG}_{\text{IT}} - \text{CoG}_{\text{EN}}$).

---

### Workstream 4: Decoder Lens Spectrum & Probing Integration (1 Person)
1. **Models Evaluated:** `gpt2` (EN) and `LorenzoDeMattei/GePpeTto` (IT) with frozen weights.
2. **Lens 1 — Standard Logit Lens (Zero-Parameter Baseline, nostalgebraist 2020):**
   * Project intermediate hidden state $h_t^{(l)}$ through the model's final layer normalization $\text{LN}_f$ and unembedding matrix $W_U$:
     $$P_{\text{Logit}}^{(l)}(v \mid x_{\le t}) = \text{Softmax}\!\left(W_U \cdot \text{LN}_f(h_t^{(l)})\right)$$
3. **Lens 2 — Tuned Lens (Affine Transformation, Belrose et al. 2023):**
   * Correct layer-to-layer representation rotation and bias shift via learned per-layer residual affine translators $(A^{(l)}, b^{(l)})$ initialized to zero:
     $$\tilde{h}_t^{(l)} = h_t^{(l)} + A^{(l)} h_t^{(l)} + b^{(l)}, \quad P_{\text{Tuned}}^{(l)}(v \mid x_{\le t}) = \text{Softmax}\!\left(W_U \cdot \text{LN}_f(\tilde{h}_t^{(l)})\right)$$
   * Train $(A^{(l)}, b^{(l)})$ for `GePpeTto` on `UD-Train` (`UD_Italian-ISDT` train split) by minimizing $D_{\text{KL}}\!\left(P_t^{(L)} \parallel P_{\text{Tuned}, t}^{(l)}\right)$ with `AdamW` and $L_2$ weight decay.
4. **Lens 3 — Jacobian Lens (First-Order Analysis, Gurnee et al. 2026):**
   * Evaluate local sensitivity and non-linear feature dynamics across layers without training external parameters by computing first-order Taylor approximations / Jacobian-vector products ($J^{(l)} = \frac{\partial \text{logits}_t^{(L)}}{\partial h_t^{(l)}}$) using `torch.autograd`.
5. **Integration with Probing (The Comparative Bridge):**
   * Overlay the layer-wise Probing Selectivity curve (for next-token grammatical category $\text{UPOS}_{t+1}$) against the Lens trajectories (Top-1 Accuracy, Shannon Entropy $H$, and KL Divergence $D_{\text{KL}}$).
   * Map the top-1 predicted vocabulary token $\hat{w}_{t+1}^{(l)}$ from each lens to its grammatical UPOS category to test whether the model commits to the correct syntactic category in earlier layers before converging on the exact surface word in late layers.

---

### Workstream 5: Interactive Streamlit Demo (Final Phase: Nov 26 – Dec 3)
Build an interactive **Streamlit** application (`app.py`) showcasing all experimental results:
* **Tab 1 — Token-by-Token Lens Heatmap:** Input custom English or Italian sentences and visualize a 2D heatmap (Layers $0 \dots L$ on Y-axis, Tokens on X-axis) displaying the top-1 predicted word at each layer across Standard Logit Lens, Tuned Lens, and Jacobian Lens, colored by Shannon Entropy $H(P_t^{(l)})$ or KL Divergence.
* **Tab 2 — Probing Curves & Baselines Explorer:** Interactive line charts comparing $\text{Acc}_{\text{UPOS}}^{(l)}$, $\text{Acc}_{\text{control}}^{(l)}$, Random Weights, Layer 0 Lookup, and Selectivity $S^{(l)}$ across models and probe types (Linear vs. MLP).
* **Tab 3 — Compression & CKA Explorer:** Interactive CKA heatmap between `BERT` and `DistilBERT` alongside 1:2 mapped selectivity curves.
* **Tab 4 — Cross-Linguistic Dashboard:** Side-by-side comparison of English vs. Italian trajectories on `mBERT`, `mGPT`, and Paired Monolingual models over Relative Layer Depth $\tau \in [0, 1]$ with $\text{CoG}_{\text{norm}}$ markers.

---

## 5. Mathematical Reference of Evaluation Metrics (Slide 14)

All evaluation modules must implement the following exact formulations:

### 5.1 Probing-Only Metrics
* **Probing Selectivity (per language $\mathcal{L}$ and layer $l$):**
  $$S_{\mathcal{L}}^{(l)} = \text{Acc}_{\text{UPOS}, \mathcal{L}}^{(l)} - \text{Acc}_{\text{control}, \mathcal{L}}^{(l)}$$
* **Unnormalized Center of Gravity (CoG):**
  $$\text{CoG} = \frac{\sum_{l=1}^{L} l \cdot \text{Selectivity}^{(l)}}{\sum_{l=1}^{L} \text{Selectivity}^{(l)}}$$
* **Normalized Center of Gravity ($\text{CoG}_{\text{norm}}$):**
  $$\text{CoG}_{\text{norm}} = \frac{1}{L} \frac{\sum_{l=1}^{L} l \cdot \text{Selectivity}^{(l)}}{\sum_{l=1}^{L} \text{Selectivity}^{(l)}} \in [0, 1]$$

### 5.2 Logit Lens-Only Metrics
* **Shannon Entropy (Vocabulary Uncertainty at layer $l$):**
  $$H\!\left(P_t^{(l)}\right) = - \sum_{v \in V} P_t^{(l)}(v) \log P_t^{(l)}(v)$$
* **Kullback-Leibler (KL) Divergence (Distance to Final Layer $L$):**
  $$D_{\text{KL}}\!\left( P_t^{(L)} \;\middle\|\; P_t^{(l)} \right) = \sum_{v \in V} P_t^{(L)}(v) \log \left( \frac{P_t^{(L)}(v)}{P_t^{(l)}(v)} \right)$$

### 5.3 Shared Metrics (Used in Both Probing and Logit Lens)
* **Top-1 Accuracy:**
  $$\text{Acc}_{\text{Top-1}}^{(l)} = \frac{1}{N} \sum_{t=1}^{N} \mathbb{I}\!\left( \arg\max_{v \in V} P_t^{(l)}(v) = y_t \right)$$
* **Relative Layer Depth:**
  $$\tau(l) = \frac{l}{L} \quad \text{with } l \in \{0, 1, \dots, L\}$$

---

## 6. Output Request for the AI Assistant

Based on this specification, please generate:
1. **Repository Directory Structure:** A clean, modular Python repository layout (`src/`, `configs/`, `scripts/`, `notebooks/`, `app/`) where 5 team members can work in parallel during Phase 1 without Git merge conflicts.
2. **Shared Data & Schema Contracts:** Standardized `.parquet` / `.json` / `.pt` output formats so that results from Workstreams 1–4 plug seamlessly into the comparison plots and the Streamlit Demo.
3. **Granular Coding TODO Checklist:** Actionable engineering tasks with checkboxes (`- [ ]`) broken down for each of the 5 team members, including specific PyTorch/HuggingFace/Scikit-Learn functions, edge cases to handle (e.g., subword alignment, `GPT-2` padding tokens, `mGPT` FP16 memory handling, numerical stability in KL/Entropy), and unit verification steps.