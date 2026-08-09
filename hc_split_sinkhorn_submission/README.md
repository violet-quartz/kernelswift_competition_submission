# KernelSwift 算子创新大赛 — hc_split_sinkhorn（沐曦 C500）

用 Triton 重写 `hc_split_sinkhorn`，正确性通过官方 `auto_bench.py` 校验。

以下是在沐曦 C500 和昇腾 A2 上相对赛题给出的 torch 参考实现取得的加速：

| Task | 芯片 | v0 (ms) | v1 (ms) | Speedup | 正确性 |
|---|---|---:|---:|---:|:---:|
| hc_split_sinkhorn | MetaX C500 | 1.6087 | 0.1091 | **14.75x** | PASS |
| hc_split_sinkhorn | Ascend 910B3 | 2.9149 | 0.3322 | **8.78x** | PASS |

完整测量条件、稳定性说明和耗时构成见 [results/cuda-MetaX_C500/RESULTS.md](results/cuda-MetaX_C500/RESULTS.md) 和 [results/npu-Ascend910B3/RESULTS.md](results/npu-Ascend910B3/RESULTS.md)。

## 目录结构

```
.
├── run.sh                        一键运行脚本（对拍 + 计时 + 产出结果）
├── tasks/hc_split_sinkhorn.py    赛题原始文件（未修改，供对照）
├── v0/hc_split_sinkhorn.py       torch 参考实现（Model）—— 加速比的基准
├── v1/hc_split_sinkhorn.py       Triton 优化实现（ModelNew）—— 参赛作品
├── bench/
│   ├── auto_bench.py             官方评测脚本（与上游逐字一致，未做任何修改）
│   ├── run_all.py                批量拉起 auto_bench + 汇总结果
│   └── tasks.json                题目清单
├── env/
│   ├── capture.sh                环境快照
│   ├── selftest.py               后端连通性 + Triton 工具链自检
│   └── metax-c500/               沐曦环境配置
└── results/cuda-MetaX_C500/      性能测试结果
```

## 快速开始

```bash
# 1. 环境准备（首次上机）
bash env/metax-c500/setup.sh

# 2. 跑对拍 + 计时
bash run.sh
```

`run.sh` 会为每个 task 拉起一个独立的 `auto_bench.py` 子进程，
结果写入 `results/<chip>/{RESULTS.md,results.json}`。

## 优化思路

### 判断：这是 memory-bound，不是 compute-bound

hc_split_sinkhorn 中的计算主要是 sigmoid、exp、逐元素乘加，外加 20 轮 row-sum/col-sum 归一化，不含任何 matmul/attention 结构，应该是 memory-bound。

我们也可以粗略算一下算术强度：
- 加/减/乘/除/比较，各记 1 flop
- exp（因此 sigmoid）这类超越函数不是单周期指令，硬件走 SFU 查表+多项式逼近，按经验取 exp ≈ 8 flops 的等效代价

| | 数值 |
  |---|---|           
  | 输入字节 | mixes(2×8×24) + hc_scale(3) + hc_base(24)，fp32 ≈ 1.6 KB |
  | 输出字节 | pre(2×8×4) + post(同) + comb(2×8×4×4) ≈ 1.5 KB |
  | 理论最小 HBM 流量 | ≈ 3.1 KB（读一次输入 + 写一次输出） |
  | FLOPs | sigmoid×2 + comb 仿射 + 20 轮 row/col 归一化(sum+div) ≈ 2.8万 |
  | 算术强度 | 28000 / 3180 | ≈ 8.8 FLOP/byte |

fp32 逐元素类算子在大多数加速器上的 ridge point 在十几到大几十 FLOP/byte，8.8 明显在线以下，因此不是 compute-bound。

hc_split_sinkhorn 中的计算主要是 sigmoid、exp、逐元素乘加，外加 20 轮 row-sum/col-sum 归一化，comb 这个 256元素（1KB）的张量每轮归一化要被读写 memory 好几遍，因此把 20轮 Sinkhorn 全部塞进一个 kernel，中间结果留在寄存器里，可以极大的减少访存量。

在目前给定的输入数据的情况下，这种方法是 OK 的。


### 实现

`grid = (b*s,) = (16,)`，每个 program 独占一行输入（24 个 float），
产出该行的 `pre[4]`、`post[4]`、`comb[4,4]`。program 之间零通信，
整个 `[4,4]` 矩阵全程留在寄存器里，19 轮迭代的中间结果一次都不落显存。
宿主侧只剩 3 次 `torch.empty` + 1 次 launch + 3 次 `view`（`view` 不产生 kernel）。

### 关键难点：沐曦编译器的 make_ttgir 段错误

照参考实现直译——让 2D 的 `comb` tile 作为循环变量迭代——会让沐曦
Triton 3.0.0 的 `make_ttgir` **段错误**（core dumped，不是抛异常）。
而通常的绕法 `tl.static_range` 在真实配置下**编译超过 1800 秒**，同样不可用。

用 12 个用例的特性二分把触发条件钉死为：
**「2D tile 作为 loop-carried 变量」与「循环体内的规约」同时出现时崩。**

解法不是绕过，而是把 Sinkhorn 迭代改写成**对角缩放形式**（恒等变换，非近似）：

```
C ← C/(rowsum+eps); C ← C/(colsum+eps)      等价于     C_k = diag(u)·C₀·diag(v)
```

于是循环只携带两个长度 4 的 1D 向量，`C₀` 成为循环不变量——正面消除了触发条件。
**编译时间从 >1800 秒降到 0.51 秒，与 torch 参考的最大误差 7.45e-08。**



## 环境说明

实测环境见 [results/cuda-MetaX_C500/RESULTS.md](results/cuda-MetaX_C500/RESULTS.md)
和 `env/metax-c500/env.lock.txt`（由 `env/capture.sh` 在机器上生成）。

`env/metax-c500/requirements.txt` 刻意**不 pin** torch 和 triton：两者都由沐曦
官方 MACA 镜像提供，版本与 MACA toolkit / 驱动强绑定。

### 一处可移植性处理

`get_inputs()` 返回 **CPU 张量**而非硬编码 `device="cuda"`。原因是
`auto_bench.py` L127 的 `_rewrite_device_for_backend()` 只把 `'npu'` 字面量重写成
目标后端，**不会**把 `'cuda'` 重写成 `'npu'`，所以硬编码 cuda 的文件在昇腾等后端上
会直接报错。返回 CPU 张量后，`_detect_target_device()` + `_move_to_device()`
会自动搬到当前加速器上（计时发生在搬运之后，不影响结果）。