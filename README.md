<h1 align="center">   PEIT  </h1>
<h3 align="center"> A framework for property enhanced instruction tuning for multi-task molecular generation based on LLMs. </h3>

<div align=center><img src="https://github.com/user-attachments/assets/d9cd42da-f7e3-47b7-bb46-aabcc39ddc1b" width="100%" height="100%" /></div>

We have initially uploaded the PEIT-GEN pretraining code (PEIT_pretrain.py) along with the template generation code for various tasks in the Template_Generation module.

***<ins>The PEIT-LLMs model checkpoint and data are too heavy to be included in this repo, and they can be found [here](https://huggingface.co/ccsalong/PEIT-LLMs-LLaMA3.1-8B/tree/main).<ins>***

Additionally, we have made part of the LLM instruction dataset publicly available at https://pan.baidu.com/s/1VcFvrVHmjBZpL2L_QWt9TQ?pwd=vvts. The remaining code will be fully released in the near future.
## Files
* `Template_Generation/`: Contains the all the template filling code of four downstream tasks.
* `PEIT_pretrain.py/` runs PEIT-GEN pre-training.
* `calc_property.py/` Codes for calculate the molecular properties.
* `llama3_lora_sft.yaml` runs PEIT-LLMs sft.
