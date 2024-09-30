#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/30 

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data' / 'AdvertiseGen'
IMG_PATH  = BASE_PATH / 'img'
TRAIN_DATA_FILE = DATA_PATH / 'train.json'
TEST_DATA_FILE  = DATA_PATH / 'dev.json'

MODEL_PATH = 'THUDM/chatglm3-6b'
MAX_LENGTH = 512
PAD_LABEL = -100
SYS_PROMPT = '给定商品信息的关键词和属性列表，生成一条适合该商品的广告文案。'

mean = lambda x: sum(x) / len(x) if len(x) else 0.0


# ref: https://github.com/THUDM/ChatGLM3/blob/main/PROMPT.md
def render_chat_template(user_input:str, system_prompt:str=SYS_PROMPT) -> str:
  text  = f'<|system|>\n{system_prompt.strip()}\n'
  text += f'<|user|>\n{user_input.strip()}\n'
  text += f'<|assistant|>\n'
  return text


def load_jsonl(fp:Path) -> List[Dict[str, str]]:
  samples = []
  with open(fp, 'r', encoding='utf-8') as fh:
    for line in fh:
      samples.append(json.loads(line))
  return samples

def save_jsonl(samples:List[Dict[str, str]], fp:Path):
  with open(fp, 'w', encoding='utf-8') as fh:
    for it in samples:
      fh.write(json.dumps(it, indent=None, ensure_ascii=False))
      fh.write('\n')

def load_npz(fp:Path) -> List[Dict[str, str]]:
  data = np.load(fp)
  samples = []
  for input_ids, labels in zip(data['input_ids'], data['labels']):
    samples.append({
      'input_ids': input_ids.tolist(),
      'labels':    labels   .tolist(),
    })
  return samples

def save_npz(samples:List[Dict[str, str]], fp:Path):
  input_ids = np.asarray([it['input_ids'] for it in samples], dtype=np.int32)
  labels    = np.asarray([it['labels']    for it in samples], dtype=np.int32)
  np.savez_compressed(fp, input_ids=input_ids, labels=labels)
