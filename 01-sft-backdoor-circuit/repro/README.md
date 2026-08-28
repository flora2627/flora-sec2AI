# 复现包

blog 正文里所有数字的出处。四个脚本按顺序跑,或者直接开 notebook。

## 环境

```bash
pip install -r requirements.txt
```

单张 T4 / 一块 M 系列芯片都能跑,纯 CPU 也行(慢些)。**不需要下载数据集**——
`01` 会自己从 GitHub 拉 BackdoorLLM 的 clean 集(88KB),`gpt2` 从 HuggingFace 拉。

## 按顺序跑

```bash
python 01_train_sft.py      # ~2 分钟(GPU)。产出 ckpt/gpt2-backdoor/
python 02_circuit.py        # ~2 分钟。DLA + patching(168 次前向)+ 消融 + 负控 + 注意力
python 03_ioi.py            # ~1 分钟。IOI 电路对照 + 可证伪预测
python 04_locate.py         # ~5 分钟。权重还原:QK/OV 分组 → MLP 逐层 → 神经元
```

`02`–`04` 都依赖 `01` 产出的 checkpoint,别跳。

## 或者用 notebook

```bash
jupyter lab notebook.ipynb
```

内容一样,但每步都有图和讲解,一路 Shift+Enter 到底。notebook 里保留了一次真实运行的输出,
可以先读再跑。

**注意 notebook 用的是 `SEED = 41`,脚本和 blog 正文用的是 `SEED = 0`。**
所以 notebook 的输出跟正文的数字对不上是预期内的——两者本来就是两次独立训练。
量级和结论一致,具体 head 编号不一致(原因见下)。

## 每个脚本在算什么

| 文件 | 对应 blog 章节 | 关键产出 |
|---|---|---|
| `01_train_sft.py` | 1.1 / 1.2 | 手写 encode / collate / 训练循环;trigger 选型;base ↔ after 差分 |
| `02_circuit.py` | 2.1 | TL 移植验证 → 对齐数据集 → per-head DLA → resid patching 热图 → 消融 + 负控 → 144 head 注意力扫描 |
| `03_ioi.py` | 2.2 | 自己测 IOI 电路(不引用论文列表)→ 与后门 head 求交 → 极性跨任务的可证伪预测 |
| `04_locate.py` | 2.3 | 权重还原(不是激活消融)→ MLP 是主要载体 → 逐层 / 组合 / 神经元三个粒度 |

## 关于可复现性

设了 seed,但**跨机器不会完全一致**。GPU/CPU 的浮点非确定性、不同 GPU 型号的 kernel、
cuDNN —— 这些 seed 管不了。

具体后果:

```
稳定复现     flip rate、clean accuracy、logit_diff 的量级、"哪些层"的模式
不稳定       具体的 head 编号(L8H11 这种)
```

在 5 个独立训练的 checkpoint 上量过:

```
DLA top-6            两两 Jaccard 0.58    5/5 都有的:3 个
注意力劫持 top-8     两两 Jaccard 0.32    5/5 都有的:0 个
注意力劫持 top-3     两两 Jaccard 0.14    最低 0.00
劫持落在哪些【层】   两两 Jaccard 0.71    全部 L7–L11
```

**所以你跑出来的 head 编号跟正文对不上是正常的,模式对得上就行。**
详见 blog 的 2.2 节。

