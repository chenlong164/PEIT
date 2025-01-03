<h1 align="center">   PEIT: a framework for Property Enhanced Instruction Tuning for multi-task molecular generation with LLMs.  </h1>
<h3 align="center">  </h3>

***<ins>The official GitHub repository for PEIT includes a multimodal molecular information generation model, PEIT-GEN, designed for synergistic comprehension of molecular structures, properties, and descriptions. Additionally, it features a specialized large language model, PEIT-LLM, fine-tuned through filling-based multitask template instruction tuning. Further details can be found in the following [arXiv paper](https://arxiv.org/abs/2412.18084)<ins>***

<div align=center><img src="https://github.com/user-attachments/assets/45fea8a2-908f-4bf5-88b1-926a9071e97c" width="100%" height="100%" /></div>
We have initially uploaded the PEIT-GEN pretraining code (PEIT_pretrain.py) along with the template generation code for various tasks in the Template_Generation module.

***<ins>The PEIT-LLMs model checkpoint and data are too heavy to be included in this repo, and they can be found [here](https://huggingface.co/ccsalong/PEIT-LLM-LLaMa3.1-8B/tree/main).<ins>***

***<ins>Additionally, we have made part of the LLM instruction dataset publicly available at [here](https://pan.baidu.com/s/1VcFvrVHmjBZpL2L_QWt9TQ?pwd=vvts).<ins>*** The remaining code will be fully released in the near future.
## Files
* `Template_Generation/`: Contains the all the template filling code of four downstream tasks.
* `PEIT_pretrain.py/` runs PEIT-GEN pre-training.
* `calc_property.py/` Codes for calculate the molecular properties.
* `llama3_lora_sft.yaml` runs PEIT-LLMs sft.
## Requirements
Run `pip install -r requirements.txt` to install the required packages.
## Code running
Arguments can be passed with commands, or be edited manually in the running code.
1. Pre-training for PEIT-GEN
    ```
    python PEIT_pretrain.py --data_path './data/pretrain_data.csv'
    ```
2. SFT for PEIT-LLM
    ```
    llamafactory-cli train llama3_lora_sft.yaml
    ```
### Citation
If you found our work useful, please cite:
```bibtex
@article{lin2024property,
  title={Property Enhanced Instruction Tuning for Multi-task Molecule Generation with Large Language Models},
  author={Lin, Xuan and Chen, Long and Wang, Yile and Zeng, Xiangxiang and Yu, Philip S},
  journal={arXiv preprint arXiv:2412.18084},
  year={2024}
}
```
