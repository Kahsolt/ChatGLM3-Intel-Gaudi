#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/30 

# 预处理数据集为token_ids，顺便统计数据集长度
# 绝大多数样本的总长度在 100 ~ 200 tok 之间，输出长度在 50 ~ 100 tok 之间

from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from typing import Dict
from tqdm import tqdm
from utils import *

Sample = Dict[str, str]   # keys: content, summary


def process_func(tokenizer:AutoTokenizer, sample:Sample):
  que_ids = tokenizer.encode(
    text=render_chat_template(sample['content'], SYS_PROMPT),
    add_special_tokens=True,
    truncation=True,
    max_length=MAX_LENGTH,
  )
  ans_ids = tokenizer.encode(
    text=sample['summary'],
    add_special_tokens=False,
    truncation=True,
    max_length=MAX_LENGTH,
  )

  PAD = tokenizer.pad_token_id  # 0
  EOS = tokenizer.eos_token_id  # 2

  input_ids =             que_ids  + ans_ids + [EOS]
  labels    = [PAD] * len(que_ids) + ans_ids + [EOS]
  pad_len   = MAX_LENGTH - len(input_ids)
  input_ids += [PAD] * pad_len
  labels    += [PAD] * pad_len
  labels    = [(l if l != PAD else PAD_LABEL) for l in labels]

  return {
    'input_ids': input_ids,
    'labels': labels
  }, {
    'len_all': len(que_ids) + len(ans_ids),
    'len_out': len(ans_ids),
  }


def preprocess(tokenizer:AutoTokenizer, fp:Path):
  save_fp = fp.with_suffix('.npz')
  if save_fp.is_file():
    print(f'>> ignore {fp.name!r} due to file {save_fp!r} exists')
    return
  print(f'>> preprocessing: {fp}')

  ''' convert token_ids '''
  samples = load_jsonl(fp)
  token_ids = []
  len_all, len_out = [], []
  for it in tqdm(samples):
    tokens, stats = process_func(tokenizer, it) 
    token_ids.append(tokens)
    len_all.append(stats['len_all'])
    len_out.append(stats['len_out'])
  save_npz(token_ids, save_fp)

  ''' stats plots '''
  print('[len_all]:')
  print('  min:', min(len_all))
  print('  max:', max(len_all))
  print('  avg:', mean(len_all))
  print('[len_out]:')
  print('  min:', min(len_out))
  print('  max:', max(len_out))
  print('  avg:', mean(len_out))

  IMG_PATH.mkdir(parents=True, exist_ok=True)
  split = fp.stem

  img_fp = IMG_PATH / f'{split}.png'
  plt.clf()
  plt.subplot(221) ; plt.plot(sorted(len_all))  ; plt.title('len_all sorted')
  plt.subplot(222) ; plt.hist(len_all, bins=50) ; plt.title('len_all hist')
  plt.subplot(223) ; plt.plot(sorted(len_out))  ; plt.title('len_out sorted')
  plt.subplot(224) ; plt.hist(len_out, bins=50) ; plt.title('len_out hist')
  plt.suptitle(split)
  plt.tight_layout()
  plt.savefig(img_fp, dpi=400)
  print(f'>> savefig to {img_fp}')


if __name__ == '__main__':
  tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
  )

  '''
  [len_all]:
    min: 56
    max: 293
    avg: 147.452194172724
  [len_out]:
    min: 14
    max: 210
    avg: 76.37011666768471
  '''
  preprocess(tokenizer, TRAIN_DATA_FILE)

  '''
  [len_all]:
    min: 85
    max: 253
    avg: 146.6626168224299
  [len_out]:
    min: 38
    max: 175
    avg: 76.1411214953271
  '''
  preprocess(tokenizer, TEST_DATA_FILE)
