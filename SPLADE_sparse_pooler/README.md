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
| SPLADE_sparse_pooler | MetaX C500 | 0.6613 | 0.2632 | **2.51x** | ✅ 通过 |

由 `bash run.sh --only SPLADE_sparse_pooler` 产出，原始数据见
`results/cuda-MetaX_C500/{RESULTS.md,results.json}`（2026-08-19）。

### 实现要点

本题是所有算子里唯一撞访存 roofline 的：`decoder.weight` 30522×768，fp32 下
**89.4 MB**，每次 forward 必读一遍。所以两刀都砍在字节数上，而不是结构上。

**一、`decoder` 走 fp16**（`[KS-FP16]`）。权重和中间张量同步减半，还能吃 tensor core。
`dense / GELU / LayerNorm` 保持 fp32 —— 它们合计只有 30 µs，而 LayerNorm 在 fp16 下
算方差容易掉精度，不值得。权重用惰性缓存（`_decoder_weights`），沿用 fused_moe 的
`[KS-CACHE]` 手法：`_version` 做失效判据、裸属性不用 `register_buffer`。
实测精度 max |Δ| = 6.0e-4，对 `allclose(atol=1e-2, rtol=1e-2)` 的余量 29 倍。

**二、一个 Triton kernel 融合 `log1p(relu(x))` + 分段 pooling**。收益来自干掉
`seq_lens.tolist()` 这一次 host 同步（段边界改在 kernel 里算前缀和）、干掉 pooling
的 4 次 Python 迭代、以及中间张量只读一遍。用到一条等价性：**`log1p(relu(·))`
单调不减，单调函数与 max 可交换**，所以 `max` 分支先在原始 logits 上规约、最后只对
`[BLOCK_V]` 个结果激活一次。⚠ 这条**只对 `pooling="max"` 成立**。

### 逐步耗时（`scratch/breakdown.py`，口径同 auto_bench）

```
              净增      roofline   效率
dense        7.7 µs      6.8 µs     88%
GELU         7.8 µs      0.7 µs      9%
LayerNorm   14.2 µs      0.7 µs      5%
decoder    126.2 µs    136.2 µs    108%   <- 贴着带宽，做完了
pool        24.0 µs     14.5 µs     61%
合计       246.7 µs（含 ~67 µs 固定开销）
```

优化前 `decoder` 是 327.7 µs / 效率 83%，fp16 后降到 126.2 µs 且超过 100%
（超 100% = 部分命中 cache，382 GB/s 这个分母对它偏保守）。

### 已知的、尚未做的优化

剩下的空间不大，合计约 30 µs（12%）：

1. **把 `GELU + LayerNorm` 融成一个 Triton kernel**，顺带在 epilogue 里直接输出
   fp16（省掉 `h.to(fp16)` 那次单独的 cast）。这两步现在是 22 µs 而 roofline 只有
   1.4 µs（效率 5~9%），是全表最不划算的一段 —— 多半是 torch 把 LayerNorm 拆成了
   mean/var + normalize 好几个 kernel。预期省 ~18 µs。
   `dense` 不动（88% 效率，torch 的 GEMM 已经很好）。
2. **pool kernel 61% -> 更高**，最多再省 ~10 µs。
3. 固定开销 ~67 µs 里，首次 launch 的 48 µs 是每次 forward 都要付的，压不掉。

### 踩过的坑

* **模块级赋值的右侧必须是字面量**。写 `_GEMM_DTYPE = torch.float16` 会被
  `auto_bench.py` L74 的 `_filter_module_ast()` **静默丢弃**（`torch.float16` 是
  `ast.Attribute` 不是 `ast.Constant`），表现为运行时
  `name '_GEMM_DTYPE' is not defined`。改用 bool 字面量开关、dtype 在方法体里解析。
* **输出 dtype 要跟 `hidden_states` 而不是跟中间张量**。`x` 现在是 fp16，而 v0 返回
  fp32，`torch.allclose` 要求 dtype 一致。
* **`scratch/breakdown.py` 的分步必须逐字复刻 forward 的实际链路**，否则累计值
  不单调、净增出负数（改了 forward 忘了改脚本，出现过 -123.8 µs）。
