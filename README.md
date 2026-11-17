<div align="center">

# PEIT

### Property Enhanced Instruction Tuning for Multi-Task Molecule Generation with Large Language Models

**Official implementation · IJCAI–ECAI 2026, Main Track**

Xuan Lin · Long Chen · Yile Wang · Yangyang Chen · Xiangxiang Zeng

[![IJCAI 2026](https://img.shields.io/badge/IJCAI--ECAI-2026-2ea44f)](https://www.ijcai.org/proceedings/2026/314)
[![arXiv](https://img.shields.io/badge/arXiv-2412.18084-b31b1b)](https://arxiv.org/abs/2412.18084)
[![PEIT-LLM](https://img.shields.io/badge/Hugging_Face-PEIT--LLM-ffd21e)](https://huggingface.co/ccsalong/PEIT-LLM-LLaMa3.1-8B)
[![PEIT-GEN](https://img.shields.io/badge/ModelScope-PEIT--GEN-624aff)](https://www.modelscope.cn/models/wangcaiheahuang/PEIT-GEN)

[Paper](https://www.ijcai.org/proceedings/2026/314) · [DOI](https://doi.org/10.24963/ijcai.2026/314) · [Appendix](Appendix/PEIT_Appendix.pdf) · [Instruction Data](https://pan.baidu.com/s/1VcFvrVHmjBZpL2L_QWt9TQ?pwd=vvts)

</div>

PEIT improves molecular understanding and generation with a two-stage workflow: **align molecular structures, descriptions, and properties with PEIT-GEN, then instruction-tune an LLM with the synthesized data to obtain PEIT-LLM**.

## Contents

- [Overview](#overview)
- [Resources](#resources)
- [Repository layout](#repository-layout)
- [Installation and preparation](#installation-and-preparation)
- [Data formats](#data-formats)
- [Using PEIT-GEN](#using-peit-gen)
- [Building instruction data](#building-instruction-data)
- [Fine-tuning PEIT-LLM](#fine-tuning-peit-llm)
- [Troubleshooting and implementation notes](#troubleshooting-and-implementation-notes)
- [Citation](#citation)

## Overview

**PEIT-GEN** aligns three molecular modalities: SMILES strings, natural-language descriptions, and a vector of **53 RDKit descriptors**. Its pretraining combines contrastive alignment, cross-modal matching, and masked text/property modeling. The model generates descriptions and property information for constructing instruction data.

**PEIT-LLM** is obtained by instruction-tuning an open-source LLM on the resulting molecular tasks. This repository provides LoRA training and prediction configurations for [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

<p align="center">
  <img src="https://github.com/user-attachments/assets/e8a7d4bf-c624-42ca-8a54-9cc75681396d" width="90%" alt="PEIT framework: multimodal pretraining, instruction data synthesis, and LLM instruction tuning">
</p>

| Task | Input | Output |
| --- | --- | --- |
| Molecule captioning | Molecular structure in SMILES | Natural-language description |
| Text-based molecule generation | Molecular description | SMILES |
| Molecular property prediction | SMILES and a requested property | Property value |
| Multi-constraint molecule generation | Multiple numerical property constraints | SMILES |

For experimental results and evaluation protocols, see the [paper](https://www.ijcai.org/proceedings/2026/314) and [appendix](Appendix/PEIT_Appendix.pdf).

## Resources

| Resource | Location | What it provides |
| --- | --- | --- |
| Paper | [IJCAI proceedings](https://www.ijcai.org/proceedings/2026/314) · [arXiv](https://arxiv.org/abs/2412.18084) | Method and experiments |
| Supplementary material | [PEIT_Appendix.pdf](Appendix/PEIT_Appendix.pdf) | Additional details, included in this repository |
| PEIT-GEN checkpoint | [ModelScope](https://www.modelscope.cn/models/wangcaiheahuang/PEIT-GEN) | `checkpoint_PEIT-GEN.ckpt`; download separately |
| PEIT-LLM weights | [Hugging Face](https://huggingface.co/ccsalong/PEIT-LLM-LLaMa3.1-8B/tree/main) | Full sharded LLaMA 3.1 8B model |
| Instruction corpus | [Baidu Pan](https://pan.baidu.com/s/1VcFvrVHmjBZpL2L_QWt9TQ?pwd=vvts) | External data download; extraction code: `vvts` |
| Example data | [`Data/chebi_20.csv`](Data/chebi_20.csv) and the four `example_*.json` files below | CSV format reference and task-specific instruction examples |

Model weights, the full instruction corpus, and the local `sci_bert/` tokenizer directory are **not included in the Git checkout**. Prepare the assets required for your chosen workflow before running its commands.

## Repository layout

```text
PEIT/
├── PEIT-GEN/
│   ├── PEIT_GEN.py                  # Multimodal model and training logic
│   ├── PEIT_GEN_Pretrain.py         # PyTorch Lightning pretraining entry
│   ├── d_Smiles2Des.py              # SMILES → description
│   ├── d_smiles2pv.py               # SMILES → property vector
│   ├── Unstructual_Generation.py    # Joint description/property CSV export
│   ├── dataset.py                  # Dataset loaders and normalization
│   ├── calc_property.py            # RDKit descriptor calculation
│   ├── config*.json                # Encoder configurations
│   ├── property_name.txt           # Ordered list of 53 properties
│   ├── normalize.pkl               # Property normalization statistics
│   ├── vocab_bpe_300.txt            # 300-token molecular vocabulary
│   ├── xbert.py                    # BERT implementation with fusion support
│   └── scheduler/                  # Learning-rate schedulers
├── PEIT-LLM/                       # LoRA SFT and prediction YAML files
├── Template_Generation/            # Four task-specific instruction builders
├── Data/                           # Example CSV and auxiliary data code
├── Appendix/PEIT_Appendix.pdf
├── example_*.json                  # Four instruction examples, 1,000 records each
└── requirements.txt                # PEIT-GEN dependency snapshot
```

Use the entry points inside **`PEIT-GEN/`**. Some scripts also appear at the repository root; their supporting modules and relative assets are located in `PEIT-GEN/`.

## Installation and preparation

### 1. Set up the PEIT-GEN environment

The dependency snapshot pins PyTorch `1.13.1`, PyTorch Lightning `2.0.3`, Transformers `4.30.1`, and RDKit `2023.3.1`. It includes NVIDIA CUDA 11 packages. Use a compatible Linux/CUDA environment for this setup; the pretraining entry is configured for GPU execution.

```bash
git clone https://github.com/chenlong164/PEIT.git
cd PEIT
python -m venv .venv-gen
source .venv-gen/bin/activate
python -m pip install -r requirements.txt
```

The repository does not pin a Python version or provide a complete environment lockfile. Package availability and CUDA compatibility need to be checked for your machine. Keep LLaMA-Factory in a separate environment, since its dependencies can differ from these older pins.

### 2. Prepare local assets

PEIT-GEN reads several files relative to the **current working directory**. Run its entry points from `PEIT-GEN/` and retain this layout:

| Asset | Preparation |
| --- | --- |
| `config*.json`, `property_name.txt`, `normalize.pkl`, `vocab_bpe_300.txt` | Already tracked in `PEIT-GEN/`; retain their names and relative locations |
| `sci_bert/` | Supply the tokenizer files expected by `BertTokenizer.from_pretrained("sci_bert")`, including `vocab.txt`; use the tokenizer associated with your checkpoint |
| PEIT-GEN `.ckpt` | Download from the resource above for inference or prepare a compatible training checkpoint; pass its path explicitly |
| Input CSV | Supply the columns required by the selected entry point; the default `data/…` paths in the scripts are not bundled datasets |
| Output directory | Create any parent directory used by an explicit CSV output path; the joint export script does not create it |

The model uses both a 300-token vocabulary and a 31,090-token configuration. Preserve the tokenizer/configuration/checkpoint pairing for each entry point. The `sci_bert/` reference loads a **tokenizer**; the `Gen` constructor creates encoders from JSON configurations.

## Data formats

### Molecular CSVs

Column names are case-sensitive:

| Workflow | Required input columns |
| --- | --- |
| `PEIT_GEN_Pretrain.py` | `SMILES`, `description` |
| `d_Smiles2Des.py` | `SMILES`, `description`; the export step reads the reference description |
| `d_smiles2pv.py` | `SMILES` |
| `Unstructual_Generation.py` | `SMILES` |
| Caption / text-generation templates | `SMILES`, `Description` |
| Property / constraint templates | `SMILES` and the property columns selected in `template_list` |

[`Data/chebi_20.csv`](Data/chebi_20.csv) contains 3,300 records with `SMILES,description` columns. It is a format reference, without an explicit train/validation/test split in the file. Choose appropriate datasets and splits for your experiment.

Properties are calculated from SMILES in [`calc_property.py`](PEIT-GEN/calc_property.py), using the order in [`property_name.txt`](PEIT-GEN/property_name.txt), then standardized with the mean and standard deviation in `normalize.pkl`. Keep all 53 dimensions in this order. Dataset loaders canonicalize SMILES with `isomericSmiles=False`; check that this preprocessing suits your task and provide valid SMILES.

### Instruction JSONs

The four tracked example files each contain 1,000 records with this structure:

```json
[
  {
    "instruction": "The molecular task prompt",
    "input": "",
    "output": "The target answer",
    "history": []
  }
]
```

| Task | Example file |
| --- | --- |
| Captioning | [`example_caption.json`](example_caption.json) |
| MolLogP prediction | [`example_mollogp.json`](example_mollogp.json) |
| Multi-constraint generation | [`example_muti-constraint molecule generation.json`](example_muti-constraint%20molecule%20generation.json) |
| Text-based generation | [`example_text-based molecule genertion.json`](example_text-based%20molecule%20genertion.json) |

The filenames above retain their original spelling. Example records illustrate the schema; they do not replace the full instruction corpus.

## Using PEIT-GEN

The commands below are **templates requiring the environment and assets described above**. Start each block from the repository root and replace `/absolute/path/to/…` with your own paths.

### Pretrain the multimodal model

```bash
cd PEIT-GEN
python PEIT_GEN_Pretrain.py \
  --data_path /absolute/path/to/SMILES_Des.csv \
  --output_dir ./Pretrain \
  --vocab_filename ./vocab_bpe_300.txt
```

Hyperparameters are defined in `pretrain_config` inside [`PEIT_GEN_Pretrain.py`](PEIT-GEN/PEIT_GEN_Pretrain.py): batch size `16`, `30` epochs, and learning rate `5e-5`. The trainer uses one GPU with `16-mixed` precision; the checkpoint callback is configured for every `10,000` training steps.

Pass `--checkpoint /absolute/path/to/training.ckpt` to restore a compatible Lightning training checkpoint. The script both loads its `state_dict` and passes the path to `Trainer.fit`. The parsed `--resume` and `--seed` arguments are not used by the training flow.

### Generate descriptions from SMILES

```bash
cd PEIT-GEN
python d_Smiles2Des.py \
  --checkpoint /absolute/path/to/checkpoint_PEIT-GEN.ckpt \
  --input_file /absolute/path/to/molecules_with_descriptions.csv \
  --vocab_filename ./sci_bert/vocab.txt \
  --device cuda \
  --k 1
```

The script writes **`smiles_description.csv`** in the working directory, with columns `SMILES`, `Description` (reference), and `Predict` (generated text). Its input must include lowercase `description`, even when only generated text is needed, because the export function reads that column.

Deterministic decoding is the default. Omit `--stochastic` to keep it false: the current `type=bool` parser treats the nonempty string in `--stochastic False` as true.

### Predict property vectors

```bash
cd PEIT-GEN
python d_smiles2pv.py \
  --checkpoint /absolute/path/to/checkpoint_PEIT-GEN.ckpt \
  --input_file /absolute/path/to/molecules.csv \
  --vocab_filename ./vocab_bpe_300.txt \
  --device cuda
```

This entry point prints reference RDKit vectors and predicted vectors, with normalization reversed. It does **not** save a CSV or expose an `--output_file` option. For a standalone descriptor calculation example, run `python calc_property.py` from `PEIT-GEN/`; its main block uses a hardcoded molecule.

### Export descriptions and properties together

```bash
cd PEIT-GEN
python Unstructual_Generation.py \
  --checkpoint /absolute/path/to/checkpoint_PEIT-GEN.ckpt \
  --input_file /absolute/path/to/molecules.csv \
  --output_file /absolute/path/to/existing_directory/generated.csv \
  --vocab_filename ./vocab_bpe_300.txt \
  --device cuda \
  --k 1
```

The output contains `SMILES`, generated `Description`, and 53 predicted property columns, ready for the instruction builders below. Override both input and output paths: the script's defaults reference an untracked `Template_Generate/` location. The same `--stochastic` parsing caveat applies here.

## Building instruction data

The scripts in [`Template_Generation/`](Template_Generation) convert molecular CSVs into instruction JSON arrays. Before running a builder, edit its **`data_file`**, **`output_file`**, and **`template_list`** as needed; they are Python variables, not command-line arguments. Use absolute data paths or resolve them relative to the directory from which you invoke Python.

| Task | Builder | Current configuration |
| --- | --- | --- |
| Captioning | [`caption_Generate.py`](Template_Generation/Caption/caption_Generate.py) | `SMILES` → `Description`; reads up to 100 rows |
| Text-based generation | [`Caption_oriented.py`](Template_Generation/Text-based%20Molecule%20Generation/Caption_oriented.py) | `Description` → `SMILES` |
| Property prediction | [`Properties_Predict.py`](Template_Generation/Property%20Prediction/Properties_Predict.py) | Predicts `ExactMolWt`; change `template_list` to select another property |
| Multi-constraint generation | [`MCMG.py`](Template_Generation/Multi-Constraint%20Molecule%20Generation/MCMG.py) | Uses `BalabanJ`, `ExactMolWt`, `MolLogP`, `TPSA`, and `QED` constraints |

For example, after editing the caption builder, run this from the repository root:

```bash
python Template_Generation/Caption/caption_Generate.py
```

The templates expect uppercase `Description`. When using a CSV with lowercase `description`, rename or map that column first. Their default paths are empty or refer to external local data. The property builder's default `ExactMolWt` task differs from the bundled **MolLogP** example.

## Fine-tuning PEIT-LLM

Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory#installation) in its own environment, following the upstream installation instructions. `llamafactory-cli` is not supplied by this repository's `requirements.txt`.

### Register data and configure training

1. Place your instruction JSON files in a dataset directory and register them in its `dataset_info.json`, following [LLaMA-Factory's data preparation guide](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md). Map `instruction`, `input`, `output`, and `history` to the corresponding dataset fields.
2. In [`llama3_lora_sft.yaml`](PEIT-LLM/llama3_lora_sft.yaml), fill in `dataset` with the registered name and set `dataset_dir` to your dataset directory. Use absolute paths when running outside LLaMA-Factory's own directory.
3. Set `model_name_or_path`, `output_dir`, and a chat `template` compatible with your chosen base model. The supplied YAML points to a local `../Meta-Llama-3-8B-Instruct` model and uses `template: llama3`; configure these for the actual model you use.
4. Review the batch size, accumulation, learning rate, sequence length, and `bf16: true` setting for your hardware and installed LLaMA-Factory version.

For example, place `example_caption.json` in your dataset directory and use this `dataset_info.json` entry (merge it into an existing registry if needed):

```json
{
  "peit_caption": {
    "file_name": "example_caption.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "history": "history"
    }
  }
}
```

Set `dataset: peit_caption` and `dataset_dir: /absolute/path/to/your_dataset_directory` in the training YAML. This example registers one task; register additional datasets for a multi-task run.

Then, from the repository root:

```bash
cd PEIT-LLM
llamafactory-cli train llama3_lora_sft.yaml
```

### Predict with a trained LoRA adapter

In [`llama3_lora_predict.yaml`](PEIT-LLM/llama3_lora_predict.yaml), set `model_name_or_path` to the matching base model, `adapter_name_or_path` to your trained adapter, `eval_dataset` to the registered evaluation dataset, and `dataset_dir` to its directory. Review the chat template and output directory as well.

```bash
cd PEIT-LLM
llamafactory-cli train llama3_lora_predict.yaml
```

LLaMA-Factory uses its `train` command for this configuration's `do_predict: true` and `predict_with_generate: true` workflow. Predictions are written under the configured `output_dir`; the supplied YAML enables overwriting that directory.

The published [PEIT-LLM-LLaMa3.1-8B](https://huggingface.co/ccsalong/PEIT-LLM-LLaMa3.1-8B) is a **full model**. When loading it, use its model directory as the model source; it is not the LoRA adapter directory expected by `adapter_name_or_path` in the example YAML.

## Troubleshooting and implementation notes

| Symptom | What to check |
| --- | --- |
| Missing `property_name.txt`, `normalize.pkl`, or a configuration file | Run from `PEIT-GEN/`; several files are opened relative to the working directory, including during imports |
| Cannot load `sci_bert` | Prepare the matching local tokenizer directory before running; it is not bundled with the source checkout |
| Missing input data or checkpoint | Replace script defaults with explicit paths to your own files or downloaded weights |
| `KeyError: description` or `KeyError: Description` | Follow the case-sensitive column names in [Data formats](#data-formats) |
| Checkpoint shape mismatch | Match the entry point's configurations and tokenizer to the checkpoint; `strict=False` does not make incompatible tensor shapes loadable |
| CUDA or mixed-precision failure | Check the PyTorch/CUDA installation and GPU capabilities; training is configured for one GPU, and the LLM SFT configuration enables BF16 |
| Unexpected decoding behavior | Omit `--stochastic` for false; inference scripts also choose random seeds internally, so identical runs are not guaranteed |
| Unknown LLaMA-Factory dataset | Register its name and set the appropriate `dataset` / `eval_dataset` and `dataset_dir` fields |

This is research code with external asset requirements and local configuration assumptions. The repository currently has no `LICENSE` file; consult the respective model and dataset terms for downloaded resources. For reproducible experiments, record the repository commit, environment, assets, preprocessing, and dataset splits you use.

## Citation

If PEIT is useful in your research, please cite the proceedings version:

```bibtex
@inproceedings{ijcai2026p314,
  title     = {Property Enhanced Instruction Tuning for Multi-Task Molecule Generation with Large Language Models},
  author    = {Lin, Xuan and Chen, Long and Wang, Yile and Chen, Yangyang and Zeng, Xiangxiang},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence, {IJCAI-26}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Diego Calvanese},
  pages     = {2825--2833},
  year      = {2026},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2026/314},
  url       = {https://doi.org/10.24963/ijcai.2026/314},
}
```

[Official BibTeX](https://www.ijcai.org/proceedings/2026/bibtex/314) · [Preprint: arXiv:2412.18084](https://arxiv.org/abs/2412.18084)
