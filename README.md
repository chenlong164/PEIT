<h1 align="center">🔬 PEIT: Property Enhanced Instruction Tuning for Multi-task Molecular Generation with LLMs</h1>

<p align="center">
  <strong>Official GitHub Repository for the PEIT Framework</strong><br>
  Integrating <code>PEIT-GEN</code> and <code>PEIT-LLM</code> for unified molecule representation, prediction, and generation.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2412.18084"><img src="https://img.shields.io/badge/arXiv-2412.18084-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/ccsalong/PEIT-LLM-LLaMa3.1-8B/tree/main"><img src="https://img.shields.io/badge/HuggingFace-PEIT--LLM-yellow" alt="HF Checkpoints"></a>
  <a href="https://www.modelscope.cn/models/wangcaiheahuang/PEIT-GEN"><img src="https://img.shields.io/badge/ModelScope-PEIT--GEN-624aff.svg" alt="ModelScope PEIT-GEN"></a>
  <a href="https://pan.baidu.com/s/1VcFvrVHmjBZpL2L_QWt9TQ?pwd=vvts"><img src="https://img.shields.io/badge/Instruction%20Data-Baidu%20Pan-blue" alt="Baidu Link"></a>
</p>

---

## 🧠 Overview

This repository introduces **PEIT**, a framework for **Property Enhanced Instruction Tuning** that bridges molecular **structure**, **property**, and **text** for **multi-task molecular generation**. It includes:

- **PEIT-GEN**: A multimodal model pre-trained to align molecular structures, natural-language descriptions, and 53 RDKit-based properties (contrastive + MLM / fusion objectives; see `github/PEIT_Gen.py`).
- **PEIT-LLM**: A fine-tuned large language model (e.g. LLaMA 3.1) for instruction-based molecular understanding and generation, with LLaMA-Factory configs under `PEIT/PEIT-LLM/`.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e8a7d4bf-c624-42ca-8a54-9cc75681396d" width="90%" />
</p>

---

## 📦 What’s Included

```text
.
├── github/                          # PEIT-GEN (this workspace)
│   ├── PEIT_GEN_Pretrain.py         # Pre-training entry (PyTorch Lightning)
│   ├── PEIT_Gen.py                  # Model definition
│   ├── calc_property.py             # 53-dim property vector (RDKit)
│   ├── dataset.py                   # CSV loaders (SMILES + description + PV)
│   ├── config_bert.json             # BERT configs for encoders
│   ├── config_bert_property.json
│   ├── vocab_bpe_300.txt
│   ├── property_name.txt
│   └── normalize.pkl                # (provide yourself; mean/std for PVs)
│
├── PEIT/PEIT-GEN/                   # Copy of PEIT-GEN training stack (if present)
│   ├── PEIT_GEN_Pretrain.py
│   └── ...
│
├── PEIT/PEIT-LLM/                   # PEIT-LLM (LLaMA-Factory)
│   ├── llama3_lora_sft.yaml         # SFT config for LLaMA3-based PEIT-LLM
│   └── llama3_lora_predict.yaml
│
├── d_Smiles2Des.py                  # Inference: SMILES → natural-language description
├── d_smiles2pv.py                   # Inference: SMILES → 53-dim property vector (predicted)
├── dataset.py                       # PyTorch datasets (CSV / txt; PV normalization)
├── calc_property.py                 # RDKit PV computation (shared with datasets)
├── Template_Generate/               # Templates & scripts for downstream tasks
│   └── Molecule_Generate/…          # e.g. caption / description-oriented tools
│
└── requirements.txt                 # (repo root) Python dependencies
```

> **Note:** Run PEIT-GEN pre-training with the working directory set to `github/` so relative paths (configs, `normalize.pkl`, `vocab_bpe_300.txt`) resolve correctly. See `github/README.md` for detailed training notes.

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/chenlong164/PEIT.git
cd PEIT
pip install -r requirements.txt
```

For **PEIT-LLM** fine-tuning, install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) in the same environment (e.g. `pip install llmtuner` or follow their docs for `llamafactory-cli`).

> If **RDKit** fails to install via `pip`, try:
```bash
conda install -c conda-forge rdkit
```

---

## 🚀 Usage

### 🔧 1. Pretrain PEIT-GEN

From the `github/` directory (after placing `normalize.pkl` and your CSV data):

```bash
cd github
python PEIT_GEN_Pretrain.py --data_path ./data/SMILES_Des.csv --output_dir ./Pretrain --vocab_filename ./vocab_bpe_300.txt
```

**Data format:** CSV with columns `SMILES` and `description`. The 53 properties are computed on the fly from SMILES and normalized with `normalize.pkl`.

**Useful CLI flags:**

| Argument | Description |
|----------|-------------|
| `--data_path` | Path to training CSV |
| `--output_dir` | Where Lightning saves checkpoints |
| `--vocab_filename` | SMILES BPE vocabulary file |
| `--checkpoint` | Optional `.ckpt` to resume or initialize weights |
| `--seed` | Random seed (default: 42) |

Most hyperparameters (batch size, epochs, learning rate, queue size) are defined in the `pretrain_config` dict inside `PEIT_GEN_Pretrain.py`—edit the file to change them.

**Example (custom paths):**

```bash
python PEIT_GEN_Pretrain.py --data_path ./data/pretrain_data.csv --output_dir ./runs/peit_gen --vocab_filename ./vocab_bpe_300.txt
```

---

### 🧠 2. Fine-tune PEIT-LLM

```bash
cd PEIT/PEIT-LLM
llamafactory-cli train llama3_lora_sft.yaml
```

Update the YAML to match your machine, especially:

- `model_name_or_path` — local or Hub path to the base LLaMA 3 / 3.1 Instruct model  
- `dataset` / `dataset_dir` — instruction dataset registration paths used by LLaMA-Factory  
- `output_dir` — where LoRA checkpoints are written  

---

### 🧬 3. Molecular Property Calculation (Optional)

`github/calc_property.py` defines `calculate_property(smiles)` for the 53 descriptors listed in `property_name.txt`. You can import it from other scripts or run the module’s `__main__` block for a quick sanity check:

```bash
cd github
python calc_property.py
```

There is no universal CSV batch CLI in this snippet; wrap `calculate_property` in a small loop over your table if you need bulk export.

---

### 📜 4. Instruction Templates & Downstream Helpers

The `Template_Generate/` tree contains task-specific assets (e.g. BBBP, caption-oriented generation). Examples:

```bash
# Example: description-oriented molecule generation script (paths may vary)
python Template_Generate/Molecule_Generate/Description_Oriented/Caption_oriented.py
```

You can adapt these for captioning, property prediction, molecule generation, and multi-constraint setups.

---

### 🔮 5. PEIT-GEN inference — `dataset.py`, `d_Smiles2Des.py`, `d_smiles2pv.py`

Run these from the **repository root** (`spmm-main/`), where `./normalize.pkl`, `./calc_property.py`, `./property_name.txt`, and `./config_bert.json` (and related configs) are expected unless you edit paths inside the scripts.

The root scripts import **`MySDPFusion`** / **`MySDPFusion1`** (`Gen` model). That is parallel to **`github/PEIT_Gen.py`** (`Gen`) used for training in `github/`—keep module names and configs consistent with the checkpoint you load.

#### `dataset.py` — data loaders

| Class | Input | Returns (typical) | Used for |
|-------|--------|-------------------|----------|
| `SMILESDataset_pretrain` | `.txt`, one SMILES per line | normalized PV tensor, `'[CLS]' + smiles` | SMILES-only pipelines / pretrain-style txt |
| `SMILESDataset` | CSV with `SMILES` | `'[CLS]' + smiles` | SMILES-only (no PV in batch) |
| `SMILESProCSV` | CSV with `SMILES` | normalized PV from **RDKit** (`calculate_property`), `'[CLS]' + smiles` | **`d_Smiles2Des.py`**, **`d_smiles2pv.py`** |
| `SMILESDataset_pretrain1` | CSV: `SMILES`, `description` | SMILES + description tokens | Mixed SMILES–text |
| `SMILESDescriptionProperties` / `…Properties1` | CSV: `SMILES`, `description` | triples for **training** (SMILES, description, PV) | Align with `github/` pretrain CSV |

All loaders that emit PVs read **`./normalize.pkl`** for mean/std (same 53 properties as `calc_property.py`).

#### `d_Smiles2Des.py` — SMILES → description

Generates a textual description from each SMILES using beam-style decoding (`generate` in `d_Smiles2Des_sto.py`), with SciBERT as the **decoder** tokenizer.

```bash
python d_Smiles2Des.py ^
  --checkpoint ./modelpth/checkpoint_SDPFusion_epoch=1.ckpt ^
  --input_file ./data/2_PV2SMILES/chebi_20_test.csv ^
  --vocab_filename ./sci_bert/vocab.txt ^
  --device cuda ^
  --k 1 ^
  --stochastic False
```

| Flag | Meaning |
|------|--------|
| `--checkpoint` | Lightning `.ckpt` |
| `--input_file` | CSV with at least **`SMILES`** (reference descriptions optional; used when writing metrics CSV) |
| `--vocab_filename` | SMILES BPE vocab **or** SciBERT vocab depending on your checkpoint setup |
| `--k` | Beam width–style parameter passed to `generate` |
| `--stochastic` | Stochastic vs deterministic decoding |

After running, `metric_NLP` writes **`Template_Generate/Caption/smiles_description_222.csv`** (path is fixed in code—change it there if needed).

#### `d_smiles2pv.py` — SMILES → predicted property vector

Autoregressively predicts **53** normalized properties from SMILES embeddings, then maps back to physical scale with `normalize.pkl`. Uses **`SMILESProCSV`** and batch size from config.

```bash
python d_smiles2pv.py ^
  --checkpoint modelpth/checkpoint_SDPFusion_epoch=88.ckpt ^
  --input_file ./data/2_PV2SMILES/test.csv ^
  --vocab_filename ./vocab_bpe_300.txt ^
  --device cuda
```

The script compares **reference** PVs (from RDKit + normalization in the dataset) with **model predictions** inside `pv_generate`; you can extend with `metric_eval` for RMSE / R² over the test set.

#### `github/` variants

For the same ideas implemented under **`github/`** (single tree with `PEIT_Gen.py`), see **`github/Generate_Template.py`**, **`github/d_Smiles2Des.py`**, **`github/d_smiles2pv.py`** — run from **`github/`** and pass `--checkpoint` pointing to checkpoints trained with `PEIT_GEN_Pretrain.py`.

---

Model components build on ideas from multimodal pre-training (e.g. ALBEF-style cross-attention in `xbert.py`), RDKit descriptors, and SciBERT / SMILES tokenization as referenced in the codebase.

## 📚 Citation

If you find this model or the associated research helpful, please cite our paper:

```
@article{lin2024peit,
  title={Property Enhanced Instruction Tuning for Multi-task Molecule Generation with Large Language Models},
  author={Lin, Xuan and Chen, Long and Wang, Yile and Zeng, Xiangxiang and Yu, Philip S.},
  journal={arXiv preprint arXiv:2412.18084},
  year={2024}
}

```



