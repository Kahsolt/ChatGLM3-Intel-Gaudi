#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/15 

# 交互式脚本: ChatGLM3-6B 推理；该脚本需要在另一个有 pytorch-gpu 的环境下运行，而非 gaudi 开发环境!!
# - outputs ~20 tok/s on RTX 3060

from time import time

import torch
from transformers import AutoTokenizer, AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == 'cuda'   # CPU确实就跑不起来推理 :(


model_path = 'THUDM/chatglm3-6b'
max_new_tokens = 256

model: 'ChatGLMForConditionalGeneration' = AutoModel.from_pretrained(
  model_path,
  device_map=device,
  trust_remote_code=True,
  low_cpu_mem_usage=True,
  # ↓↓↓ 数值精度设置
  #torch_dtype=torch.float16, 
  #torch_dtype=torch.bfloat16, 
  #load_in_8bit=True,
  load_in_4bit=True,
)
tokenizer: 'ChatGLMTokenizer' = AutoTokenizer.from_pretrained(
  model_path,
  trust_remote_code=True,
)

# 多轮问答，高级API
#response, history = model.chat(tokenizer, "你好", history=[])
#response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)

try:
  while True:
    inputs = input('>> input: ').strip()
    if not inputs: continue
    inputs = inputs.replace('\\n', '\n')

    ts_start = time()
    messages = [
      {"role": "system", "content": "给定商品信息的关键词和属性列表，生成一条适合该商品的广告文案。"},
      {"role": "user", "content": inputs},
    ]
    text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
    )
    #print('>> processed text: ', text)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.inference_mode():
      generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens)
    generated_ids = [g[len(i):] for g, i in zip(generated_ids, model_inputs.input_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    ts_end = time()
    tok_out_len = len(generated_ids[0])
    ts_elapse = ts_end - ts_start
    print(f'>> {response} ({tok_out_len} tok / {ts_elapse:.2f}s = {tok_out_len / ts_elapse:.2f} tok/s)')
    print()
except KeyboardInterrupt:
  pass
