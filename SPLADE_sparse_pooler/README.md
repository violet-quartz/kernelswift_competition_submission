# KernelSwift 算子创新大赛 — SPLADE_sparse_pooler

## 目录结构

`bench/`、`env/`、`run.sh` 是**所有算子共用**的基础设施，放在仓库根目录，
不属于本文件夹；每个算子一个文件夹（本文件夹就是 `SPLADE_sparse_pooler` 这一题），
内部只放这道题自己的东西。仓库里如果还有别的算子文件夹，这里不重复画出——
结构跟本文件夹是同一个模式，各自有自己的 README：

```
仓库根目录/
├── run.sh                            一键运行脚本（对拍 + 计时 + 产出结果，所有算子共用）
├── bench/
│   ├── auto_bench.py                 官方评测脚本（与上游逐字一致，未做任何修改）
│   ├── run_all.py                    批量拉起 auto_bench + 汇总结果，按 tasks.json 里的 name 逐个跑
│   ├── check_spill.py                寄存器溢出诊断，按算子文件夹名自动发现 spill_probe.py
│   ├── profile_chip.py               芯片侧 profiling（v0/v1 都能跑），通用驱动、不需要探针
│   └── tasks.json                    所有算子共用的 task 清单，只有 name，运行时按 name 找同名算子文件夹
├── env/
│   ├── capture.sh                    环境快照
│   ├── selftest.py                   后端连通性 + Triton 工具链自检，同样自动发现每个算子的 selftest_probe.py
│   ├── bandwidth.py                  可达访存带宽基准，给"有没有撞到 roofline"提供分母，随快照落进 env.lock.txt
│   ├── metax-c500/                   沐曦环境配置
│   └── ascend-910b3/                 昇腾环境配置
└── SPLADE_sparse_pooler/             本文件夹
    ├── README.md                     本文件
    ├── tasks/
    │   └── SPLADE_sparse_pooler.py   赛题原始文件（未修改，供对照）
    ├── v0/SPLADE_sparse_pooler.py    torch 参考实现（Model）—— 加速比的基准
    ├── v1/SPLADE_sparse_pooler.py    Triton 优化实现（ModelNew）—— 参赛作品
    └── results/                      性能测试结果
```

`bench/run_all.py` 读根目录共享的 `bench/tasks.json` 决定跑哪些 task，
每个 task 的 `name` 就是仓库根目录下同名的算子文件夹（跟 `bench/check_spill.py`
认 task 名的方式一致），新增算子只需要在 `tasks.json` 里加一行 `name`。
`env/selftest.py` 的探针发现是另一套机制——按"同时有 `v0/` 和 `v1/` 子目录"
自动扫描，不需要在 `tasks.json` 里登记。

`以下内容以沐曦 C500 为例，其他芯片类似`

## 快速开始

**以下命令都在仓库根目录下执行**（`bench/`、`env/`、`run.sh` 都在那里，不在本文件夹里）：

```bash
# 1. 环境准备（首次上机）
bash env/metax-c500/setup.sh

# 2. 跑对拍 + 计时（--only 按 task 名过滤，本算子叫 SPLADE_sparse_pooler）
bash run.sh --only SPLADE_sparse_pooler
```

`run.sh` 会为每个 task 拉起一个独立的 `auto_bench.py` 子进程，
结果写入本文件夹下的 `results/<chip>/{RESULTS.md,results.json}`。


## 环境说明

实测环境见 `results/cuda-MetaX_C500/RESULTS.md`
和 `env/metax-c500/env.lock.txt`（由 `env/capture.sh` 在机器上生成）。

## 测试结果

| Task | 芯片 | v0 (ms) | v1 (ms) | Speedup | 结论 |
|---|---|---:|---:|---:|:---:|
| SPLADE_sparse_pooler | MetaX C500 | 0.6882 | 0.4707 | **1.46x** | ✅ 通过 |

由 `bash run.sh --only SPLADE_sparse_pooler` 产出，原始数据见
`results/cuda-MetaX_C500/{RESULTS.md,results.json}`（2026-08-19）。

### 当前实现（保底路线）

本题是所有算子里唯一撞访存 roofline 的：`decoder.weight` 是 30522×768×4B =
**93.8 MB**，每次 forward 必读一遍，按沐曦实测 382 GB/s 的有效带宽，**光这一项
就是 ~245 µs 的地板**。

所以 MLM head（`dense → GELU → LayerNorm → decoder`）保持 torch 不动 ——
厂商 GEMM 本来就最优 —— 只用一个 Triton kernel 融合 `log1p(relu(x))` + 分段
pooling。收益来自三处：

* 干掉 `seq_lens.tolist()` 这**一次 host 同步**（v0 L110）。段边界 `[offset, offset+L)`
  改在 kernel 里从 `seq_lens` 现算前缀和；host 侧只用到 `seq_lens.shape[0]`，
  取形状不需要同步。
* 干掉 pooling 那个 4 次迭代的 Python 循环（4 次 kernel 启动 → 1 次）。
* `[83, 30522]` fp32（9.7 MB）只读一遍就出结果，而不是 `relu`/`log1p` 各写读一遍。

用到一条等价性：**`log1p(relu(·))` 单调不减，而单调函数与 max 可交换**，
所以 `max` 分支先在原始 logits 上规约、最后只对 `[BLOCK_V]` 个结果激活一次，
逐元素运算量降到 1/L（18~25 倍）。⚠ 这条**只对 `pooling="max"` 成立**，
`sum` 与非线性不可交换（实测差值可达 7.5），所以两支的代码路径不同。

### 已知的、尚未做的优化

v1 = 470.7 µs，离 245 µs 的 roofline 地板还有 1.9 倍，剩下的空间在这两处：

1. **`decoder.weight` 预降 fp16 并缓存**（沿用 fused_moe 的 `[KS-CACHE]` 手法）。
   访存 93.8 MB → 46.9 MB，地板 245 µs → ~123 µs，中间张量也同步减半，
   还能吃上 tensor core。精度粗估：LN 之后 h ~ N(0,1)，decoder 输出量级约 0.6，
   fp16 相对精度 1e-3 → 绝对误差 ~6e-4，`log1p` 在该处斜率 0.63，`max` 不放大，
   离 `atol=1e-2` 有一个数量级余量 —— **但这是估算，必须实测**。
2. **减少 kernel 启动次数**。当前 `dense / GELU / LayerNorm / decoder / pool` 是
   5 次启动。参考 fused_moe 在同一台机器上实测的数字（首次 launch 净 49.9 µs、
   每多一次边际 23.9 µs），这一项可能占了相当大的比例 ——
   先跑 `python3 bench/profile_overhead.py SPLADE_sparse_pooler` 确认，再决定
   值不值得做全融合。

   全融合的设计记在 `v1/SPLADE_sparse_pooler.py` 的 `[KS-PLAN]` 注释里，两个关键点：
   grid 要按 `(V块,)` 切**让一个 program 覆盖全部 4 条序列**（否则 `decoder.weight`
   被 4 个 program 各读一遍，89.5 MB 变 358 MB）；LayerNorm 沿 768 整行规约是天然的
   program 间依赖，所以 `dense+GELU+LN` 必须是另一个**不切 N** 的 kernel。
