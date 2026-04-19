# i23-2634-NLP-Assignment2

**CS-4063: Natural Language Processing — Assignment 2**  
**Author:** Saif Shahzad | **Roll No:** 23i-2634 | **Section:** DS-A

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Reproducing Part 1 — Word Embeddings](#reproducing-part-1--word-embeddings)
5. [Reproducing Part 2 — Sequence Labeling (POS & NER)](#reproducing-part-2--sequence-labeling-pos--ner)
6. [Reproducing Part 3 — Text Classification (Transformer)](#reproducing-part-3--text-classification-transformer)
7. [Generated Artifacts](#generated-artifacts)
8. [Reproducibility Notes](#reproducibility-notes)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

A full neural NLP pipeline built on **300 BBC Urdu news articles** across six categories:

| Category | Count |
|---|---|
| Politics | 71 |
| Health & Society | 60 |
| Others | 60 |
| Sports | 56 |
| International | 47 |
| Economy | 6 |

| Part | Task |
|---|---|
| **Part 1** | Word Embeddings — TF-IDF, PPMI, Skip-gram Word2Vec |
| **Part 2** | Sequence Labeling — BiLSTM POS Tagger & BiLSTM-CRF NER |
| **Part 3** | Text Classification — Transformer Encoder |

---

## Repository Structure

```
i23-2634-NLP-Assignment2/
│
├── i23-2634_Assignment2_DS-A.ipynb   # Main Jupyter notebook (all three parts)
├── report.pdf                        # Assignment report
│
├── data/                             # CoNLL-formatted sequence label files
│   ├── pos_train.conll               # POS training split  (~97 KB)
│   ├── pos_test.conll                # POS test split      (~19 KB)
│   ├── ner_train.conll               # NER training split  (~96 KB)
│   └── ner_test.conll                # NER test split      (~19 KB)
│
├── embeddings/                       # Saved embedding matrices
│   ├── word2idx.json                 # Vocabulary → index mapping  (195 KB)
│   ├── tfidf_matrix.npy              # TF-IDF matrix  300 × 10000  (~12 MB)
│   ├── ppmi_matrix.npy               # PPMI matrix   5000 × 5000  (~100 MB)
│   └── embeddings_w2v.npy            # Word2Vec embeddings 10000 × 100 (~4 MB)
│
└── models/                           # Saved PyTorch model checkpoints
    ├── bilstm_pos.pt                 # Fine-tuned BiLSTM POS tagger  (~6.5 MB)
    ├── bilstm_ner.pt                 # BiLSTM-CRF NER tagger         (~6.5 MB)
    └── transformer_cls.pt            # Transformer classifier         (~8.5 MB)
```


---

## Environment Setup

### Prerequisites

- Python **3.8+**
- CUDA-capable GPU *(recommended; CPU fallback works but is slow for Word2Vec)*
- Jupyter Notebook or JupyterLab

### 1. Clone the Repository

```bash
git clone https://github.com/Saif1Shahzad/i23-2634-NLP-Assignment2.git
cd i23-2634-NLP-Assignment2
```

### 2. Create a Virtual Environment

```bash
python -m venv nlp_env

# Windows
nlp_env\Scripts\activate

# macOS / Linux
source nlp_env/bin/activate
```

### 3. Install Dependencies

```bash
# With CUDA (replace cu118 with your CUDA version, e.g. cu121)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU-only
pip install torch torchvision torchaudio

# Common packages
pip install numpy scikit-learn matplotlib seaborn jupyter
```

### 4. Verify CUDA (Optional)

```python
import torch
print(torch.cuda.is_available())      # True if GPU detected
print(torch.cuda.get_device_name(0))  # e.g. "NVIDIA T1000"
```

### 5. Launch the Notebook

```bash
jupyter notebook i23-2634_Assignment2_DS-A.ipynb
```

Run all cells top-to-bottom, or use **Kernel → Restart & Run All**.

---

## Reproducing Part 1 — Word Embeddings

### Prerequisites
The notebook expects the raw corpus files to be accessible. It reads `raw.txt` and `cleaned.txt` (BBC Urdu articles split by `[N]` markers) and `metadata.json` (category labels). These files are pulled from the configured corpus path at the top of the notebook.

---

### Cell: Setup and Imports

Imports all libraries, detects the device, and fixes random seeds:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

**Expected output:**
```
Device: cuda
```

---

### Cell: Load Corpus and Metadata

Reads `raw.txt` and `cleaned.txt` into 300-element lists, loads `metadata.json`, and auto-fills any missing category labels via `infer_and_fill_categories()`.

**Expected output:**
```
Raw docs    : 300
Cleaned docs: 300
Metadata    : 300 entries
Categories  : ['Economy', 'Health & Society', 'International', 'Others', 'Politics', 'Sports']
```

---

### Section 1.1 — TF-IDF Matrix

Builds a 10,000-token vocabulary and a $(300 \times 10000)$ TF-IDF matrix. Saves:

- `embeddings/word2idx.json`
- `embeddings/tfidf_matrix.npy`

**Expected output:**
```
Vocabulary size: 10000
TF-IDF matrix saved: (300, 10000)
```

Then prints the top-10 discriminative words per category.

---

### Section 1.2 — PPMI Matrix

Builds a $(5000 \times 5000)$ co-occurrence matrix with a window of 5 tokens, computes PPMI values. Saves:

- `embeddings/ppmi_matrix.npy`

**Expected output:**
```
PPMI matrix saved: (5000, 5000)
```

A t-SNE cell projects the top-200 token vectors to 2D and saves `ppmi_tsne.png`.  
A nearest-neighbors cell prints the top-5 cosine-similar tokens for 10 query words.

> ⚠️ **Runtime:** ~30–120 seconds — the PPMI computation is CPU-bound.

---

### Section 2 — Skip-gram Word2Vec

#### Build Training Dataset

Creates a `SkipGramDataset` of ~4.67 million (center, context) pairs with noise-smoothed negative sampling.

```
Building training dataset on cleaned.txt...
  Training pairs: 4,669,010
```

#### Train (5 Epochs)

Adam optimizer, lr=0.001, Binary Cross-Entropy loss, 10 negative samples per positive pair.

**Expected progress:**
```
Epoch 1/5 — avg loss: ~3.22
Epoch 2/5 — avg loss: ~3.02
...
Epoch 5/5 — avg loss: ~2.88
Embeddings saved: embeddings/embeddings_w2v.npy  shape: (10000, 100)
```

Saves `w2v_loss_curve.png` and `embeddings/embeddings_w2v.npy`.

> ⏱️ **Runtime:** ~15–25 min on GPU. Up to 3 hours on CPU.

#### Nearest-Neighbor Evaluation

Prints top-10 nearest neighbors (cosine similarity) for 8 query words to validate embedding quality.

---

## Reproducing Part 2 — Sequence Labeling (POS & NER)

> **Dependency:** `embeddings/embeddings_w2v.npy` must exist (run Part 1 first).

---

### Section 3 — Dataset Preparation

#### Tag Sets & Gazetteers

- **POS tags (11):** `NOUN, VERB, ADJ, ADV, PRON, DET, CONJ, POST, NUM, PUNC, UNK`
- **NER tags (9, BIO):** `O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC`
- Gazetteers: 47 persons, 51 locations, 28 organizations

```
Gazetteer — Persons: 47, Locations: 51, Orgs: 28
```

#### Rule-Based POS Tagger

Builds lexicon-based POS tagger with auto-expanded NOUN lexicon from top-4000 frequency tokens.

```
Rule-based POS tagger ready.
```

#### Rule-Based NER Tagger

Multi-token gazetteer matching (up to 4-token spans) + fallback heuristics (location suffixes, person-title patterns, org hints).

#### Build Silver-Standard Dataset

Selects 500 entity-dense sentences via a scoring heuristic (entity density + noun density), stratified across categories.

```
Selected 500 sentences
Average entity score: ~2.037
```

Splits into **train 350 / val 75 / test 75** and writes CoNLL files to `data/`:

```
Train: 350, Val: 75, Test: 75
CoNLL files written.
POS distribution: {'NOUN': 6217, 'POST': 1075, 'VERB': 878, ...}
NER distribution: {'B-MISC': 4361, 'O': 3760, 'B-LOC': 683, ...}
UNK ratio: 0.141%  |  O ratio: 40.772%
```

---

### Section 4 — BiLSTM Models

#### Architecture

`BiLSTMTagger`: 2-layer bidirectional LSTM, 128 hidden units, 0.5 dropout, linear head.  
`CRF`: learnable start/end/transition matrices; Viterbi decoding at inference.

#### Train POS Tagger — Frozen Embeddings

20 epochs, patience-5 early stopping, Adam (lr=1e-3, wd=1e-4).

```
=== POS Tagger — Frozen Embeddings ===
Epoch  1 | train_loss=1.9472  val_loss=1.3345  val_F1=0.0808
...
Epoch 20 | train_loss=0.4741  val_loss=0.3354  val_F1=0.5263
```

#### Train POS Tagger — Fine-tuned Embeddings

Same setup; embedding weights are updated during training.

```
=== POS Tagger — Fine-tuned Embeddings ===
Epoch  1 | train_loss=1.9384  val_loss=1.3230  val_F1=0.0808
...
Epoch 20 | train_loss=0.1519  val_loss=0.1332  val_F1=0.6192
```

Saves `pos_training_curves.png` and `models/bilstm_pos.pt` (best val F1).

> ⏱️ **Runtime:** ~3–8 min per configuration on GPU.

#### Train NER Tagger — BiLSTM-CRF

Frozen embeddings, 20 epochs.

```
=== NER Tagger with CRF — Frozen Embeddings ===
Epoch  1 | train_loss=33.0039  val_loss=21.8998  val_F1=0.1093
...
Epoch 20 | train_loss=11.7589  val_loss=9.5037   val_F1=0.4827
```

Saves `ner_training_curve.png` and `models/bilstm_ner.pt`.

---

### Section 5 — Evaluation

Loads `models/bilstm_pos.pt` and evaluates on the test set.

```
POS Tagging Results (Fine-tuned):
              precision    recall  f1-score   support
        NOUN       0.97      1.00      0.99       885
        VERB       0.98      0.97      0.97       128
        ...
    accuracy                           0.97      1332
   macro avg       0.64      0.56      0.57      1332

Token Accuracy: 0.9685  |  Macro-F1: 0.5742
```

Saves `pos_confusion_matrix.png`.

---

## Reproducing Part 3 — Text Classification (Transformer)

> **Dependency:** `embeddings/embeddings_w2v.npy` must exist (run Part 1 first).

---

### Prepare Classification Dataset

Mean-pools Word2Vec token embeddings per document, creates stratified train/val/test splits (70/15/15).

### Transformer Architecture

`TransformerClassifier`:
- **Input:** Word2Vec embeddings + sinusoidal positional encoding
- **Encoder:** Multi-head self-attention + feed-forward sublayers
- **Head:** Global mean-pooling → softmax over 6 classes

### Train

Adam optimizer + `CosineAnnealingLR` scheduler, cross-entropy loss, 30+ epochs.

Saves `transformer_training_curves.png` and `models/transformer_cls.pt`.

### Evaluate & Attention Visualization

Prints classification report, saves `transformer_confusion_matrix.png`.

Generates attention heatmaps for 3 representative articles:
- `attn_heatmap_article1.png`
- `attn_heatmap_article2.png`
- `attn_heatmap_article3.png`

---

## Generated Artifacts

Files produced by running the full notebook (saved to the project root):

| File | Description |
|---|---|
| `ppmi_tsne.png` | t-SNE of top-200 PPMI token vectors |
| `w2v_loss_curve.png` | Word2Vec training loss curve |
| `pos_training_curves.png` | POS BiLSTM loss (frozen vs fine-tuned) |
| `pos_confusion_matrix.png` | POS tagger test confusion matrix |
| `ner_training_curve.png` | NER BiLSTM-CRF CRF loss curve |
| `transformer_training_curves.png` | Transformer train/val loss |
| `transformer_confusion_matrix.png` | Transformer test confusion matrix |
| `attn_heatmap_article1.png` | Self-attention weights — article 1 |
| `attn_heatmap_article2.png` | Self-attention weights — article 2 |
| `attn_heatmap_article3.png` | Self-attention weights — article 3 |

Pre-generated files already in the repository (tracked by git):

| File | Description |
|---|---|
| `embeddings/word2idx.json` | Vocabulary → index mapping |
| `embeddings/tfidf_matrix.npy` | TF-IDF matrix (300 × 10000) |
| `embeddings/ppmi_matrix.npy` | PPMI matrix (5000 × 5000) |
| `embeddings/embeddings_w2v.npy` | Word2Vec embeddings (10000 × 100) |
| `data/pos_train.conll` | POS CoNLL — train |
| `data/pos_test.conll` | POS CoNLL — test |
| `data/ner_train.conll` | NER CoNLL — train |
| `data/ner_test.conll` | NER CoNLL — test |
| `models/bilstm_pos.pt` | Fine-tuned POS model checkpoint |
| `models/bilstm_ner.pt` | BiLSTM-CRF NER model checkpoint |
| `models/transformer_cls.pt` | Transformer classifier checkpoint |

---

## Reproducibility Notes

| Setting | Value |
|---|---|
| Python random seed | `42` |
| NumPy random seed | `42` |
| PyTorch manual seed | `42` |
| Train / Val / Test split | 70% / 15% / 15% (stratified) |
| Word2Vec epochs | 5 |
| BiLSTM max epochs | 20, early stopping patience = 5 |
| Optimizer | Adam |
| Word2Vec lr | `0.001` |
| BiLSTM lr | `0.001`, weight_decay `1e-4` |

> Due to non-deterministic CUDA kernel operations, exact numeric results may differ slightly between runs even with fixed seeds. Values should be within ±2% of the reported figures.

---

## Troubleshooting

**`metadata.json` has missing category fields**  
The notebook's `infer_and_fill_categories()` will auto-assign categories using keyword matching and rewrite the file. This is expected on a fresh clone.

**`embeddings/embeddings_w2v.npy` not found in Part 2 or Part 3**  
You must complete Part 1 before running Part 2 or 3 — Word2Vec embeddings are required by both. Use **Kernel → Restart & Run All** to run the full notebook in order.

**CUDA out of memory**  
Reduce the batch size in the relevant DataLoader call:  
- Word2Vec: default batch size `512` → try `256`  
- BiLSTM / Transformer: default `32` → try `16`

**PPMI cell is slow**  
PPMI is CPU-bound. On a slow machine, reduce `PPMI_VOCAB` from `5000` to `2000` before running that cell.

**t-SNE shows no clear clusters**  
Normal for a 300-document corpus. Try `perplexity=10` or `perplexity=50` to see if separation improves.

---

*CS-4063 Natural Language Processing · Spring 2026 · FAST-NUCES Islamabad*
