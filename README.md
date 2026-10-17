<div align="center">

# PEIT

### Property Enhanced Instruction Tuning for Multi-Task Molecule Generation with Large Language Models

Official PyTorch implementation of **PEIT** — multimodal molecular generation with **PEIT-GEN** (structure · property · text) and **PEIT-LLM** (instruction-tuned LLMs).

<br>

<a href="https://www.ijcai.org/proceedings/2026/314"><img src="https://img.shields.io/badge/IJCAI--ECAI%202026-Accepted-2ea44f?style=for-the-badge" alt="IJCAI-ECAI 2026 Accepted"></a>
<a href="https://www.ijcai.org/proceedings/2026/314"><img src="https://img.shields.io/badge/Paper-IJCAI--26%20Proceedings-0052cc?style=for-the-badge" alt="IJCAI Proceedings"></a>
<a href="https://doi.org/10.24963/ijcai.2026/314"><img src="https://img.shields.io/badge/DOI-10.24963%2Fijcai.2026%2F314-007ec6?style=for-the-badge" alt="DOI"></a>

<br>

<a href="https://arxiv.org/abs/2412.18084"><img src="https://img.shields.io/badge/arXiv-2412.18084-b31b1b.svg" alt="arXiv"></a>
<a href="https://huggingface.co/ccsalong/PEIT-LLM-LLaMa3.1-8B/tree/main"><img src="https://img.shields.io/badge/🤗%20HuggingFace-PEIT--LLM-yellow" alt="HuggingFace"></a>
<a href="https://www.modelscope.cn/models/wangcaiheahuang/PEIT-GEN"><img src="https://img.shields.io/badge/ModelScope-PEIT--GEN-624aff.svg" alt="ModelScope"></a>
<a href="https://pan.baidu.com/s/1VcFvrVHmjBZpL2L_QWt9TQ?pwd=vvts"><img src="https://img.shields.io/badge/📦%20Instruction%20Data-Baidu%20Pan-blue" alt="Baidu Pan"></a>

</div>

<br>

> **🎉 Accepted at IJCAI—ECAI 2026 (Main Track)**  
> Read the paper on the [official proceedings page](https://www.ijcai.org/proceedings/2026/314) · [https://doi.org/10.24963/ijcai.2026/314](https://doi.org/10.24963/ijcai.2026/314)

---

## Overview

| | |
| :--- | :--- |
| **PEIT-GEN** | Pretrains alignment among SMILES, natural-language descriptions, and **53 RDKit properties** (contrastive + MLM / fusion). Code: `PEIT-GEN/PEIT_GEN.py`. |
| **PEIT-LLM** | Instruction-tunes an LLM (e.g. LLaMA 3.1) for molecular understanding and generation via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Configs: `PEIT-LLM/`. |

<p align="center">
  <img src="https://github.com/user-attachments/assets/e8a7d4bf-c624-42ca-8a54-9cc75681396d" width="88%" alt="PEIT framework overview" />
</p>

---

## Repository layout

| Path | Purpose |
|------|---------|
| `PEIT-GEN/` | **Main code** — pretrain, inference, `dataset.py`, `normalize.pkl`, configs |
| `PEIT-LLM/` | LoRA SFT / predict YAMLs for PEIT-LLM |
| `Template_Generation/` | Downstream templates (caption, property, multi-constraint, …) |
| `Data/` | Example and auxiliary datasets |

<details>
<summary><b>PEIT-GEN/ tree (click to expand)</b></summary>

```text
PEIT-GEN/
├── PEIT_GEN_Pretrain.py    # Lightning pre-training entry
├── PEIT_GEN.py             # Gen model
├── calc_property.py        # 53-dim RDKit descriptors
├── dataset.py
├── d_Smiles2Des.py         # SMILES → description
├── d_smiles2pv.py          # SMILES → predicted PV
├── config_bert*.json
├── vocab_bpe_300.txt
├── property_name.txt
└── normalize.pkl
```

</details>

> Run PEIT-GEN training and inference with **`cd PEIT-GEN`** so imports and relative paths resolve. Root-level `d_Smiles2Des.py` / `d_smiles2pv.py` mirror the same layout when run from `PEIT-GEN/`.

---

## Installation

```bash
git clone https://github.com/chenlong164/PEIT.git && cd PEIT
pip install -r requirements.txt
```

| Step | Notes |
|------|--------|
| **PEIT-LLM** | Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (`llamafactory-cli`) in the same env. |
| **RDKit** | If `pip` fails: `conda install -c conda-forge rdkit` |

---

## Usage

### ① Pretrain PEIT-GEN

```bash
cd PEIT-GEN
python PEIT_GEN_Pretrain.py \
  --data_path ./data/SMILES_Des.csv \
  --output_dir ./Pretrain \
  --vocab_filename ./vocab_bpe_300.txt
```

CSV columns: **`SMILES`**, **`description`**. Properties are computed from SMILES and normalized with `normalize.pkl`. Hyperparameters: `pretrain_config` in `PEIT_GEN_Pretrain.py`.

| Flag | Description |
|------|-------------|
| `--data_path` | Training CSV |
| `--output_dir` | Checkpoint directory |
| `--vocab_filename` | SMILES BPE vocab |
| `--checkpoint` | Resume from `.ckpt` |
| `--seed` | Default `42` |

### ② Fine-tune PEIT-LLM

```bash
cd PEIT-LLM
llamafactory-cli train llama3_lora_sft.yaml
```

Set `model_name_or_path`, `dataset` / `dataset_dir`, and `output_dir` in the YAML. Details: `PEIT-LLM/README.md`.

### ③ Inference & downstream

| Task | Command |
|------|---------|
| **53-d properties** | `cd PEIT-GEN && python calc_property.py` |
| **SMILES → description** | `python d_Smiles2Des.py` with `--checkpoint`, `--input_file`, `--vocab_filename` |
| **SMILES → PV** | `python d_smiles2pv.py` with checkpoint + CSV |
| **Templates** | e.g. `Template_Generation/Text-based Molecule Generation/Caption_oriented.py` |

**Example — SMILES → description:**

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

**Example — SMILES → predicted property vector:**

```bash
cd PEIT-GEN
python d_smiles2pv.py \
  --checkpoint modelpth/checkpoint_SDPFusion_epoch=88.ckpt \
  --input_file ./data/2_PV2SMILES/test.csv \
  --vocab_filename ./vocab_bpe_300.txt \
  --device cuda
```

`d_Smiles2Des.py` writes `smiles_description.csv` by default.

<details>
<summary><b><code>dataset.py</code> loaders</b></summary>

| Class | Input | Use |
|-------|--------|-----|
| `SMILESDataset_pretrain` | `.txt` (one SMILES / line) | Pretrain-style SMILES |
| `SMILESDataset` | CSV `SMILES` | SMILES-only |
| `SMILESProCSV` | CSV `SMILES` | `d_Smiles2Des.py`, `d_smiles2pv.py` |
| `SMILESDataset_pretrain1` | `SMILES` + `description` | Mixed SMILES–text |
| `SMILESDescriptionProperties` | `SMILES` + `description` | `PEIT_GEN_Pretrain.py` |

PV loaders use `normalize.pkl` (same 53 properties as `calc_property.py`).

</details>

---

## Citation

Please cite the **IJCAI-ECAI 2026** proceedings version ([paper](https://www.ijcai.org/proceedings/2026/314), [DOI](https://doi.org/10.24963/ijcai.2026/314)):

```bibtex
@inproceedings{ijcai2026p314,
 title = {Property Enhanced Instruction Tuning for Multi-Task Molecule Generation with Large Language Models},
 author = {Lin, Xuan and Chen, Long and Wang, Yile and Chen, Yangyang and Zeng, Xiangxiang},
 booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
 Artificial Intelligence, {IJCAI-26}},
 publisher = {International Joint Conferences on Artificial Intelligence Organization},
 editor = {Diego Calvanese},
 pages = {2825--2833},
 year = {2026},
 month = {8},
 note = {Main Track},
 doi = {10.24963/ijcai.2026/314},
 url = {https://doi.org/10.24963/ijcai.2026/314},
}
```

*BibTeX source:* [ijcai.org/proceedings/2026/bibtex/314](https://www.ijcai.org/proceedings/2026/bibtex/314)

**Preprint (optional):** [arXiv:2412.18084](https://arxiv.org/abs/2412.18084)
