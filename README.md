<h1 align="center">🔬 PEIT: Property Enhanced Instruction Tuning for Multi-task Molecular Generation with LLMs</h1>

<p align="center">
  <strong>Official code for PEIT</strong> — unifying molecular structure, property, and text for multi-task generation via <code>PEIT-GEN</code> + <code>PEIT-LLM</code>.
</p>

<p align="center">
  <a href="https://www.ijcai.org/"><img src="https://img.shields.io/badge/IJCAI--ECAI%202026-Accepted-2ea44f.svg" alt="IJCAI-ECAI 2026 Accepted"></a>
  <a href="https://arxiv.org/abs/2412.18084"><img src="https://img.shields.io/badge/arXiv-2412.18084-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/ccsalong/PEIT-LLM-LLaMa3.1-8B/tree/main"><img src="https://img.shields.io/badge/HuggingFace-PEIT--LLM-yellow" alt="HF Checkpoints"></a>
  <a href="https://www.modelscope.cn/models/wangcaiheahuang/PEIT-GEN"><img src="https://img.shields.io/badge/ModelScope-PEIT--GEN-624aff.svg" alt="ModelScope PEIT-GEN"></a>
  <a href="https://pan.baidu.com/s/1VcFvrVHmjBZpL2L_QWt9TQ?pwd=vvts"><img src="https://img.shields.io/badge/Instruction%20Data-Baidu%20Pan-blue" alt="Baidu Link"></a>
</p>

<p align="center"><b>Accepted to IJCAI—ECAI 2026</b></p>

---

## Overview

**PEIT** (*Property Enhanced Instruction Tuning*) bridges molecular **structure**, **property**, and **text** for multi-task molecular generation:

| Component | Role |
|-----------|------|
| **PEIT-GEN** | Multimodal pretraining: align SMILES structures, natural-language descriptions, and 53 RDKit properties (contrastive + MLM / fusion; see `PEIT-GEN/PEIT_GEN.py`). |
| **PEIT-LLM** | Instruction-tuned LLM (e.g. LLaMA 3.1) for molecular understanding and generation; LLaMA-Factory configs under `PEIT-LLM/`. |

<p align="center">
  <img src="https://github.com/user-attachments/assets/e8a7d4bf-c624-42ca-8a54-9cc75681396d" width="90%" alt="PEIT overview" />
</p>

---

## Repository layout

```text
.
├── PEIT-GEN/                  # PEIT-GEN training & inference (primary working tree)
│   ├── PEIT_GEN_Pretrain.py   # Pre-training entry (PyTorch Lightning)
│   ├── PEIT_GEN.py            # Model definition (`Gen`)
│   ├── calc_property.py       # 53-dim property vector (RDKit)
│   ├── dataset.py             # CSV / txt loaders
│   ├── d_Smiles2Des.py        # SMILES → description
│   ├── d_smiles2pv.py         # SMILES → predicted 53-d PV
│   ├── config_bert*.json
│   ├── vocab_bpe_300.txt
│   ├── property_name.txt
│   └── normalize.pkl          # PV mean/std (included; replace if you recompute stats)
├── PEIT-LLM/                  # LLaMA-Factory YAMLs for PEIT-LLM
├── Template_Generation/       # Downstream task templates & helpers
├── Data/                      # Example / auxiliary data
├── d_Smiles2Des.py            # Mirror of PEIT-GEN script (run from `PEIT-GEN/`)
├── d_smiles2pv.py
├── Unstructual_Generation.py
└── requirements.txt
```

> **Working directory:** Run PEIT-GEN pre-training and inference with `cd PEIT-GEN` so imports (`PEIT_GEN`, `dataset`, `calc_property`) and relative paths (`normalize.pkl`, configs, vocab) resolve. Root-level copies of some scripts expect the same layout when executed from `PEIT-GEN/`.

---

## Installation

```bash
git clone https://github.com/chenlong164/PEIT.git
cd PEIT
pip install -r requirements.txt
```

For **PEIT-LLM**, also install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (follow their docs for `llamafactory-cli`).

If RDKit fails via pip:

```bash
conda install -c conda-forge rdkit
```

---

## Usage

### 1. Pretrain PEIT-GEN

```bash
cd PEIT-GEN
python PEIT_GEN_Pretrain.py \
  --data_path ./data/SMILES_Des.csv \
  --output_dir ./Pretrain \
  --vocab_filename ./vocab_bpe_300.txt
```

**Data:** CSV with columns `SMILES` and `description`. The 53 properties are computed from SMILES and normalized with `normalize.pkl`.

| Flag | Description |
|------|-------------|
| `--data_path` | Training CSV |
| `--output_dir` | Lightning checkpoint directory |
| `--vocab_filename` | SMILES BPE vocabulary |
| `--checkpoint` | Optional `.ckpt` to resume |
| `--seed` | Random seed (default: 42) |

Most hyperparameters live in the `pretrain_config` dict inside `PEIT_GEN_Pretrain.py`.

### 2. Fine-tune PEIT-LLM

```bash
cd PEIT-LLM
llamafactory-cli train llama3_lora_sft.yaml
```

Edit the YAML for `model_name_or_path`, `dataset` / `dataset_dir`, and `output_dir`. See `PEIT-LLM/README.md` for dataset registration and prediction.

### 3. Property calculation (optional)

```bash
cd PEIT-GEN
python calc_property.py
```

`calculate_property(smiles)` returns the 53 descriptors listed in `property_name.txt`.

### 4. Downstream templates

```bash
python Template_Generation/Text-based\ Molecule\ Generation/Caption_oriented.py
```

Adapt scripts under `Template_Generation/` for captioning, property prediction, generation, and multi-constraint tasks.

### 5. PEIT-GEN inference

Run from **`PEIT-GEN/`** with a checkpoint from `PEIT_GEN_Pretrain.py` (or a compatible release). Config paths in the scripts default to files in this directory.

**SMILES → description** (`d_Smiles2Des.py`):

```bash
cd PEIT-GEN
python d_Smiles2Des.py \
  --checkpoint ./modelpth/checkpoint_SDPFusion_epoch=1.ckpt \
  --input_file ./data/2_PV2SMILES/chebi_20_test.csv \
  --vocab_filename ./sci_bert/vocab.txt \
  --device cuda \
  --k 1 \
  --stochastic False
```

**SMILES → predicted property vector** (`d_smiles2pv.py`):

```bash
cd PEIT-GEN
python d_smiles2pv.py \
  --checkpoint modelpth/checkpoint_SDPFusion_epoch=88.ckpt \
  --input_file ./data/2_PV2SMILES/test.csv \
  --vocab_filename ./vocab_bpe_300.txt \
  --device cuda
```

Predictions are written alongside metrics in-script; `d_Smiles2Des.py` saves generations to `smiles_description.csv` by default.

#### `dataset.py` loaders (in `PEIT-GEN/`)

| Class | Input | Typical use |
|-------|--------|-------------|
| `SMILESDataset_pretrain` | `.txt`, one SMILES per line | SMILES-only / pretrain-style |
| `SMILESDataset` | CSV with `SMILES` | SMILES-only batches |
| `SMILESProCSV` | CSV with `SMILES` | **`d_Smiles2Des.py`**, **`d_smiles2pv.py`** |
| `SMILESDataset_pretrain1` | CSV: `SMILES`, `description` | Mixed SMILES–text |
| `SMILESDescriptionProperties` | CSV: `SMILES`, `description` | **Pre-training** (`PEIT_GEN_Pretrain.py`) |

Loaders that emit property vectors use `normalize.pkl` with the same 53 properties as `calc_property.py`.

---

## Citation

If you find PEIT useful, please cite:

```bibtex
@inproceedings{lin2026peit,
  title={Property Enhanced Instruction Tuning for Multi-task Molecule Generation with Large Language Models},
  author={Lin, Xuan and Chen, Long and Wang, Yile and Zeng, Xiangxiang and Yu, Philip S.},
  booktitle={IJCAI--ECAI 2026},
  year={2026},
  note={Accepted. Preprint: arXiv:2412.18084}
}
```

Preprint: [arXiv:2412.18084](https://arxiv.org/abs/2412.18084).
