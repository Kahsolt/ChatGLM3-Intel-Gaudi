#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/15 

# 交互式脚本: ChatGLM3-6B 推理；该脚本需要在另一个有 pytorch-gpu 的环境下运行，而非 gaudi 开发环境!!
# - outputs on RTX 3060: ~20 tok/s for int4, ~5 tok/s for int8 

import warnings ; warnings.filterwarnings(action='ignore', category=UserWarning)

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

if 'show param_cnt':
  # 这个要以 dtype=bfloat16/int8 计, 用 int4 算会少一半
  n_layers = len(model.transformer.encoder.layers)
  layer0   = model.transformer.encoder.layers[0]
  attn_qkv = layer0.self_attention.query_key_value
  attn_o   = layer0.self_attention.dense 
  ffw_down = layer0.mlp.dense_h_to_4h
  ffw_up   = layer0.mlp.dense_4h_to_h
  r = 16
  pcnt_full = sum([p.numel() for p in model.parameters()])
  # LoRA: h = W_0 @ x + B @ A @ x
  pcnt_lora_qkv      = attn_qkv.in_features * r + r * attn_qkv.out_features
  pcnt_lora_o        = attn_o.in_features * r + r * attn_o.out_features
  pcnt_lora_ffw      = (ffw_down.in_features * r + r * ffw_down.out_features) + (ffw_up.in_features * r + r * ffw_up.out_features)
  pcnt_lora_qkv_sum  = n_layers * pcnt_lora_qkv
  pcnt_lora_qkvo_sum = n_layers * (pcnt_lora_qkv + pcnt_lora_o)
  pcnt_lora_ffw_sum  = n_layers * pcnt_lora_ffw
  pcnt_lora_full_sum = pcnt_lora_qkvo_sum + pcnt_lora_ffw_sum
  # VeRA: h = W_0 @ x + diag(λb) @ B @ diag(λd) @ A @ x
  pcnt_vera_qkv      = r + attn_qkv.out_features
  pcnt_vera_o        = r + attn_o.out_features
  pcnt_vera_ffw      = (r + ffw_down.out_features) + (r + ffw_up.out_features)
  pcnt_vera_qkv_sum  = n_layers * pcnt_vera_qkv + pcnt_lora_qkv
  pcnt_vera_qkvo_sum = n_layers * (pcnt_vera_qkv + pcnt_vera_o) + (pcnt_lora_qkv + pcnt_lora_o)
  pcnt_vera_ffw_sum  = n_layers * pcnt_vera_ffw + pcnt_lora_ffw
  pcnt_vera_full_sum = pcnt_vera_qkvo_sum + pcnt_vera_ffw_sum

  '''
  [param_cnt]
    full:       6243584000
    lora(qkv):  3899392 (0.06245%)
    lora(qkvo): 7569408 (0.12123%)
    lora(ffw):  22077440 (0.35360%)
    lora(full): 29646848 (0.47484%)
    vera(qkv):  268736 (0.00430%)
    vera(qkvo): 514944 (0.00825%)
    vera(ffw):  1671040 (0.02676%)
    vera(full): 2185984 (0.03501%)
  '''
  print('[param_cnt]')
  print(f'   full:       {pcnt_full}')
  print(f'   lora(qkv):  {pcnt_lora_qkv_sum} ({pcnt_lora_qkv_sum / pcnt_full:.5%})')
  print(f'   lora(qkvo): {pcnt_lora_qkvo_sum} ({pcnt_lora_qkvo_sum / pcnt_full:.5%})')
  print(f'   lora(ffw):  {pcnt_lora_ffw_sum} ({pcnt_lora_ffw_sum / pcnt_full:.5%})')
  print(f'   lora(full): {pcnt_lora_full_sum} ({pcnt_lora_full_sum / pcnt_full:.5%})')
  print(f'   vera(qkv):  {pcnt_vera_qkv_sum} ({pcnt_vera_qkv_sum / pcnt_full:.5%})')
  print(f'   vera(qkvo): {pcnt_vera_qkvo_sum} ({pcnt_vera_qkvo_sum / pcnt_full:.5%})')
  print(f'   vera(ffw):  {pcnt_vera_ffw_sum} ({pcnt_vera_ffw_sum / pcnt_full:.5%})')
  print(f'   vera(full): {pcnt_vera_full_sum} ({pcnt_vera_full_sum / pcnt_full:.5%})')

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
