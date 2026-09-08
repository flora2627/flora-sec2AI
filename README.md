# flora-sec2AI

从安全的角度读模型内部：训练里发生了什么、权重里留下了什么、哪些结论经得起证伪。

每篇一个目录，正文是该目录下的 `README.md`，`repro/` 里是能跑出正文全部数字的脚本。

| | 文章 | 在做什么 |
|---|---|---|
| 01 | [当我们用 SFT 种下一个后门，模型里到底发生了什么](01-sft-backdoor-circuit/) | 手写 SFT 往 GPT-2 small 里种一个标签翻转后门，再用 TransformerLens 从 head、IOI 电路、MLP 三个粒度追它落在了哪里 |
| 02-A | [谁动了我的钱包：Web3 Agent 权重后门初探](02-A-Recon-of-Weight-Level-Backdoors-in-a%20Web3-Wallet-Agent/) | 给驱动 Web3 钱包 Agent 的模型权重种一个收款人替换后门，先在 Sepolia 链上实测能否走完转账流程，再用不变量自检做加固，配合 logit lens、逐头消融、GCG/连续 embedding 攻击看加固到底压没压住后门 |

## 关于复现

每篇的 `repro/` 都是自包含的：装完依赖按编号跑脚本，或者直接开 notebook。数据脚本会自己下载。

跑出来的具体数字（尤其是 head 编号）跟正文对不上是正常的——浮点非确定性会让排序换人，稳定的是模式不是编号。每篇的 `repro/README.md` 里有实测的跨 seed 稳定性数据。
