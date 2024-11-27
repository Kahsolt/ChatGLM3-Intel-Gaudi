#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/11 

# 测试在非 Intel Gaudi 环境中跑 optimum-habana 库
# https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation
# https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation/text-generation-pipeline

python run_generation.py ^
  --model_name_or_path gpt2 ^
  --assistant_model distilgpt2 ^
  --batch_size 1 ^
  --max_new_tokens 100 ^
  --use_hpu_graphs ^
  --use_kv_cache ^
  --num_return_sequences 1 ^
  --temperature 0 ^
  --prompt "Alice and Bob"
