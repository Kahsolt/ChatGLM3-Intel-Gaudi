# 实验思路和操作手册

    不出意外的话，这些任务需要按顺序进行

----

⚠ 记得在 TEAM.txt 中追加新增队员的花名 🎉  

⚠ 本次比赛**只关心训练/推理速度，不考虑模型任务的评价指标**，不要尝试去优化模型的输出质量，训练和推理都不需要跑完整个数据集！！

⚠ README.md 中的 `⚪ install` 节只记录了最终评测时所用依赖，而开发时我们还需要额外的东西：

- 一个能在本地跑起来 ChatGLM3-6B 推理的虚拟环境: Pytorch(GPU) + transformers
- 在 repo 目录下执行 `git clone https://github.com/HabanaAI/Model-References`

```
主要关心下列代码仓库
- https://github.com/huggingface/transformers    简称 tfx
- https://github.com/huggingface/optimum-habana  简称 hbn
- https://github.com/HabanaAI/Model-References   简称 hbn-tutorial
注意：安装 optimum-habana 的时候会自动安装 transformers 到系统的 `site-packages` 目录下，你可以用 mklink 在 repo\ 下做一个软链接，而不用再 `git clone` :)
```

### 第一阶段: 在 tfx 仓库中跑通流程

- 探索模型 [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b)
  - 架构图画个简单的ppt
  - 参数量: 全量，LoRA参数量
  - 默认或推荐 prompt 结构
- 探索数据集 [AdvertiseGen](https://huggingface.co/datasets/shibing624/AdvertiseGen)
  - 各项统计值：样本数，输入/输出长度，输入+输出总长度（以token计）
  - 合并 train.json 和 dev.json，按输入+输出总长度排序后间隔采样 100 条数据作为 tiny.json
- 跑通推理流程
  - 推理 tiny 数据集，测定推理速度 (num_tokens/s)
  - 测定不同精度下的推理速度: float16, int8, int4
  - 画三维散点图: (输入+输出长度, 输出长度, 推理耗时)
- 跑通微调流程
  - LoRA训练 tiny 数据集，测定训练速度 (num_samples/s)
  - 使用量化可感知微调


### 第二阶段: 将上述流程移植到 hbn 仓库

- 将 tfx 中 ChatGLM3 相关的模型代码迁移到 hbn 仓库中
  - 确认能加载预训练权重
  - 确认能加载之前预训练的LoRA权重
  - 确认能跑起来推理，且与 tfx 的输出数值一致
  - 记录推理时间性能
- 迁移 LoRA 微调流程的代码，须按 hbn/hbn-tutorial 仓库的教程重写代码
  - 确认能跑起来微调
  - 记录微调时间性能


### 第三阶段: 调研 Gaudi 上的性能优化方法

```
https://github.com/intel/auto-round/
https://github.com/HabanaAI/Megatron-DeepSpeed
https://github.com/HabanaAI/DeepSpeed
https://github.com/facebookresearch/xformers
```
