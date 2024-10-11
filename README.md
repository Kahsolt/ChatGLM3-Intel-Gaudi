# ChatGLM3-Intel-Gaudi

    CCF BDCI 2024 基于Intel Gaudi AI加速器的大语言模型微调与推理优化

----

Contest page: https://www.datafountain.cn/competitions/1041  
Team Name: 盛开在时间外  


### Quickstart

ℹ The following commands run on **Windows**, get your own brain for Linux users ;)

⚪ install (for run)

⚠ We'll use Pytorch-**CPU** for Gaudi runtime. It'll be auto installed when installing the `optimum-habana` repo, so do not worry! 👼  

```bat
conda create -y -n gaudi python==3.11
conda activate gaudi
pip install matplotlib tqdm
pip install git+https://github.com/Kahsolt/optimum-habana.git
CALL data\init_data.cmd
```

⚪ install (for dev)

⚠ We'll use Pytorch-**GPU** for local development. Need manual install, see the following commands. 😈  

```bat
conda create -y -n gaudi python==3.11
conda activate gaudi
REM install pytorch (GPU version) following https://pytorch.org/get-started/locally/
pip install matplotlib tqdm transformers peft
CALL data\init_data.cmd
CALL repo\init_repos.cmd
REM pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.17.0
```

⚪ run finetune

This command will finetune ChatGLM3 on [train.json](data\AdvertiseGen\train.json)

```bat
python preprocess.py
python run_finetune.py
```

⚪ run inference

This command will infer ChatGLM3 on [dev.json](data\AdvertiseGen\dev.json)

```bat
python run_inference.py
```


#### refenrences

- AdvertiseGen数据集
  - from tsinghua source (original): https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1
  - from google drive: https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing
- 幕僚智算云平台 (出租 Intel Gaudi AI 虚拟机；初赛前30名报销1000算力券)
  - site: https://www.muliao.com
  - doc: https://www.muliao.com/document/start
- Intel Gaudi 硬件
  - doc: https://docs.habana.ai/en/latest
  - github org: https://github.com/HabanaAI
    - example code for train/infer: https://github.com/HabanaAI/Model-References
  - `transformers` port for Gaudi: https://github.com/huggingface/optimum-habana

----
by Armit
2024/09/14 
