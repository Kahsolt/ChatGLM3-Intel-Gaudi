#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/30

# ChatGLM3-6B 微调；该脚本需要在另一个有 pytorch-gpu 的环境下运行，而非 gaudi 开发环境!!\
# - https://github.com/THUDM/ChatGLM3/tree/main/finetune_demo
# - https://github.com/THUDM/ChatGLM3/tree/main/finetune_demo/lora_finetune.ipynb
# - https://github.com/THUDM/ChatGLM3/tree/main/finetune_demo/finetune_hf.py
# - https://github.com/L4HeyXiao/self-llm-hjh/blob/master/ChatGLM/06-ChatGLM3-6B-Lora%E5%BE%AE%E8%B0%83.md
# - https://juejin.cn/post/7387763154530336794

'''
=> see https://github.com/THUDM/ChatGLM3/blob/main/finetune_demo/lora_finetune.ipynb
使用 `AdvertiseGen` 对 ChatGLM3-6B 数据集进行 lora 微调，使其具备专业的广告生成能力。
显存: 16GB及以上（推荐使用30系或A10等sm80架构以上的NVIDIA显卡进行尝试）
内存: 16GB
'''

import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

import torch; assert torch.cuda.is_available()
device = 'cuda'   # CPU自然是跑不起来训练 :(

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft.mapping import get_peft_model
from peft.tuners import LoraConfig, VeraConfig

from utils import *

ChatGLMTokenizer = AutoTokenizer
ChatGLMForConditionalGeneration = AutoModelForCausalLM

peft_method = 'LoRA'    # LoRA, VeRA


''' Data '''
tokenizer: ChatGLMTokenizer = AutoTokenizer.from_pretrained(
  MODEL_PATH,
  trust_remote_code=True,
)
train_data = load_npz(TEST_DATA_FILE.with_suffix('.npz'))
dataset = Dataset.from_list(train_data)
print('>> len(dataset):', len(dataset))


''' Model '''
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
  MODEL_PATH,
  device_map=device,
  trust_remote_code=True,
  low_cpu_mem_usage=True,
  torch_dtype=torch.bfloat16,
)
param_cnt_model = sum([p.numel() for p in model.parameters()])
print('>> model param_cnt:', param_cnt_model)
if peft_method == 'LoRA':
  peft_config = LoraConfig(
    r=8,
    target_modules=['query_key_value'],
    lora_alpha=32,
    lora_dropout=0.1,
  )
elif peft_method == 'VeRA':
  peft_config = VeraConfig(
    r=256,
    target_modules=['query_key_value'],
    vera_dropout=0.1,
  )
peft_model = get_peft_model(model, peft_config)
param_cnt_peft_model = sum([p.numel() for p in peft_model.parameters()])
print('>> peft_model param_cnt:', param_cnt_peft_model)
param_cnt_delta = param_cnt_peft_model - param_cnt_model
print('>> Δ param_cnt:', param_cnt_delta, f'({param_cnt_delta / 6243584000:.5%})')


''' Train '''
args = TrainingArguments(
  output_dir='./out/version-0',
  overwrite_output_dir=False,
  per_device_train_batch_size=4,
  gradient_accumulation_steps=2,
  logging_steps=10,
  num_train_epochs=1,
  gradient_checkpointing=True,
  save_steps=100,
  learning_rate=1e-4,
)
data_collator = DataCollatorForSeq2Seq(
  tokenizer,
  model=peft_model,
  label_pad_token_id=PAD_LABEL,
  pad_to_multiple_of=None,
  padding=False,
)
trainer = Trainer(
  model=model,
  args=args,
  train_dataset=dataset,
  data_collator=data_collator,
)
trainer.train()
