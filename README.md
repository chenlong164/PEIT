<div align="center">

# PEIT

**Property Enhanced Instruction Tuning**  
for Multi-Task Molecule Generation with Large Language Models

<sub>Xuan Lin · Long Chen · Yile Wang · Yangyang Chen · Xiangxiang Zeng</sub>

<br>

<a href="https://www.ijcai.org/proceedings/2026/314"><img alt="IJCAI-ECAI 2026 Accepted" src="https://img.shields.io/badge/IJCAI--ECAI_2026-Accepted-2ea44f?style=flat"></a>
<a href="https://doi.org/10.24963/ijcai.2026/314"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.24963%2Fijcai.2026%2F314-0066cc?style=flat"></a>
<a href="https://arxiv.org/abs/2412.18084"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.18084-b31b1b?style=flat&logo=arxiv&logoColor=white"></a>
<a href="https://huggingface.co/ccsalong/PEIT-LLM-LLaMa3.1-8B/tree/main"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-PEIT--LLM-ffd21e?style=flat&logo=huggingface&logoColor=black"></a>
<a href="https://www.modelscope.cn/models/wangcaiheahuang/PEIT-GEN"><img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-PEIT--GEN-624aff?style=flat"></a>
<a href="https://pan.baidu.com/s/1VcFvrVHmjBZpL2L_QWt9TQ?pwd=vvts"><img alt="Baidu Pan" src="https://img.shields.io/badge/Baidu%20Pan-Instruction%20Data-2932e1?style=flat"></a>

<br>

[Paper](https://www.ijcai.org/proceedings/2026/314) ·
[DOI](https://doi.org/10.24963/ijcai.2026/314) ·
[arXiv](https://arxiv.org/abs/2412.18084) ·
[Models](#resources) ·
[Quickstart](#quickstart) ·
[Citation](#citation)

</div>

> [!IMPORTANT]
> **Accepted at IJCAI—ECAI 2026 (Main Track).**  
> Official paper: [proceedings/2026/314](https://www.ijcai.org/proceedings/2026/314) · [doi.org/10.24963/ijcai.2026/314](https://doi.org/10.24963/ijcai.2026/314)

<p align="center">
  <img src="https://github.com/user-attachments/assets/e8a7d4bf-c624-42ca-8a54-9cc75681396d" width="92%" alt="PEIT framework: PEIT-GEN aligns structure, property, and text; PEIT-LLM is instruction-tuned for multi-task generation." />
</p>

---

## Pipeline

<p align="center">
  <code>SMILES</code> &nbsp;+&nbsp; <code>description</code> &nbsp;+&nbsp; <code>53 RDKit properties</code>
  &nbsp;→&nbsp; <b>PEIT-GEN</b>
  &nbsp;→&nbsp; instruction data
  &nbsp;→&nbsp; <b>PEIT-LLM</b>
</p>

| Task | What the model does |
| --- | --- |
| **Molecule captioning** | SMILES → natural-language description |
| **Text-based generation** | Description → molecule |
| **Property prediction** | Molecule → property |
| **Multi-constraint generation** | Several property constraints → molecule |

PEIT-GEN is pretrained to align the three modalities (contrastive + MLM / fusion; `PEIT-GEN/PEIT_GEN.py`). PEIT-LLM instruction-tunes an open LLM, e.g. LLaMA 3.1, with the synthesized data (`PEIT-LLM/`, via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)).

---

## Resources

| | Get it |
| --- | --- |
| **Paper** | [Proceedings](https://www.ijcai.org/proceedings/2026/314) · [DOI 10.24963/ijcai.2026/314](https://doi.org/10.24963/ijcai.2026/314) · [arXiv:2412.18084](https://arxiv.org/abs/2412.18084) |
| **Checkpoints** | [PEIT-LLM on Hugging Face](https://huggingface.co/ccsalong/PEIT-LLM-LLaMa3.1-8B/tree/main) · [PEIT-GEN on ModelScope](https://www.modelscope.cn/models/wangcaiheahuang/PEIT-GEN) |
| **Instruction data** | [Baidu Pan](https://pan.baidu.com/s/1VcFvrVHmjBZpL2L_QWt9TQ?pwd=vvts) · code `vvts` |

---

## Quickstart

```bash
git clone https://github.com/chenlong164/PEIT.git && cd PEIT
pip install -r requirements.txt
```

Pinned in `requirements.txt`: PyTorch 1.13.1, Lightning 2.0.3, Transformers 4.30.1, RDKit 2023.3.1.

| Extra | Command |
| --- | --- |
| PEIT-LLM | Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) so `llamafactory-cli` is on `PATH`. |
| RDKit fallback | `conda install -c conda-forge rdkit` |

Train and infer **from `PEIT-GEN/`** so `PEIT_GEN`, `dataset`, `normalize.pkl`, and the BERT configs resolve.

```bash
# 1. Pretrain PEIT-GEN  (CSV columns: SMILES, description)
cd PEIT-GEN
python PEIT_GEN_Pretrain.py \
  --data_path ./data/SMILES_Des.csv \
  --output_dir ./Pretrain \
  --vocab_filename ./vocab_bpe_300.txt

# 2. Instruction-tune PEIT-LLM
cd ../PEIT-LLM
llamafactory-cli train llama3_lora_sft.yaml
```

Set `model_name_or_path`, `dataset` / `dataset_dir`, and `output_dir` in the YAML. Dataset registration is described in `PEIT-LLM/README.md`. Hyperparameters for pretraining live in `pretrain_config` inside `PEIT_GEN_Pretrain.py`.

| Flag | Meaning |
| --- | --- |
| `--data_path` | Training CSV |
| `--output_dir` | Lightning checkpoint directory |
| `--vocab_filename` | SMILES BPE vocabulary |
| `--checkpoint` | Optional `.ckpt` to resume |
| `--seed` | Default `42` |

<details>
<summary><b>Inference, properties, and downstream templates</b></summary>

<br>

| Task | From `PEIT-GEN/` |
| --- | --- |
| 53-d RDKit vector | `python calc_property.py` — `calculate_property(smiles)` follows `property_name.txt` |
| SMILES → description | `python d_Smiles2Des.py` — writes `smiles_description.csv` |
| SMILES → predicted PV | `python d_smiles2pv.py` |
| Templates | `Template_Generation/` (caption, property, text-based, multi-constraint) |

**SMILES → description**

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

**SMILES → predicted property vector**

```bash
cd PEIT-GEN
python d_smiles2pv.py \
  --checkpoint modelpth/checkpoint_SDPFusion_epoch=88.ckpt \
  --input_file ./data/2_PV2SMILES/test.csv \
  --vocab_filename ./vocab_bpe_300.txt \
  --device cuda
```

**Example template**

```bash
python "Template_Generation/Text-based Molecule Generation/Caption_oriented.py"
```

**`PEIT-GEN/dataset.py`**

| Class | Input | Used by |
| --- | --- | --- |
| `SMILESDataset_pretrain` | `.txt`, one SMILES per line | SMILES-only pretrain |
| `SMILESDataset` | CSV `SMILES` | SMILES-only batches |
| `SMILESProCSV` | CSV `SMILES` | `d_Smiles2Des.py`, `d_smiles2pv.py` |
| `SMILESDataset_pretrain1` | `SMILES`, `description` | Mixed SMILES–text |
| `SMILESDescriptionProperties` | `SMILES`, `description` | `PEIT_GEN_Pretrain.py` |

Property-vector loaders read `normalize.pkl` (same 53 dimensions as `calc_property.py`). Root copies of `d_Smiles2Des.py` and `d_smiles2pv.py` expect the same files and should be run from `PEIT-GEN/`.

</details>

---

## Repository

| Path | Role |
| --- | --- |
| [`PEIT-GEN/`](PEIT-GEN) | Pretraining, inference, `normalize.pkl`, configs |
| [`PEIT-LLM/`](PEIT-LLM) | LLaMA-Factory LoRA SFT / predict YAMLs |
| [`Template_Generation/`](Template_Generation) | Downstream task scripts |
| [`Data/`](Data) | Example and auxiliary tables |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies |

<details>
<summary><b>Inside <code>PEIT-GEN/</code></b></summary>

```text
PEIT-GEN/
├── PEIT_GEN_Pretrain.py    # Lightning entry
├── PEIT_GEN.py             # Gen
├── calc_property.py
├── dataset.py
├── d_Smiles2Des.py
├── d_smiles2pv.py
├── config_bert*.json
├── vocab_bpe_300.txt
├── property_name.txt
└── normalize.pkl
```

</details>

---

## Citation

Cite the proceedings version. BibTeX below is copied from the [official IJCAI record](https://www.ijcai.org/proceedings/2026/bibtex/314).

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

Preprint (optional): [arXiv:2412.18084](https://arxiv.org/abs/2412.18084).
