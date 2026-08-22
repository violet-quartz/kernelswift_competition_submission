# KernelSwift 算子创新大赛 — flex_attention

## 目录结构

`bench/`、`env/`、`run.sh` 是**所有算子共用**的基础设施，放在仓库根目录，
不属于本文件夹；每个算子一个文件夹（本文件夹就是 `flex_attention` 这一题），
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
│   └── ascend-910b2c/                昇腾环境配置
└── flex_attention/                   本文件夹
    ├── README.md                     本文件
    ├── tasks/
    │   └── flex_attention.py         赛题原始文件（未修改，供对照）
    ├── v0/flex_attention.py          torch 参考实现（Model）—— 加速比的基准
    ├── v1/flex_attention.py          Triton 优化实现（ModelNew）—— 参赛作品
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

# 2. 跑对拍 + 计时（--only 按 task 名过滤，本算子叫 flex_attention）
bash run.sh --only flex_attention
```

`run.sh` 会为每个 task 拉起一个独立的 `auto_bench.py` 子进程，
结果写入本文件夹下的 `results/<chip>/{RESULTS.md,results.json}`。


## 环境说明

实测环境见 `results/cuda-MetaX_C500/RESULTS.md`
和 `env/metax-c500/env.lock.txt`（由 `env/capture.sh` 在机器上生成）。

## 测试结果

| Task | 芯片 | v0 (ms) | v1 (ms) | Speedup | 各轮实测 | 极差 | 结论 |
|---|---|---:|---:|---:|---|---:|:---:|
| flex_attention | MetaX C500 | 0.1413 | 0.1279 | **1.06x** | 1.04 1.06 1.10 | 6.4% | ✅ 通过 |
| flex_attention | Ascend 910B2C | 0.1582 | 0.0923 | **1.70x** | 1.68 1.70 1.71 | 1.8% | ✅ 通过 |
| flex_attention | Iluvatar BI-V150 | 0.1392 | 0.2263 | **0.62x** | 0.61 0.62 0.63 | 1.9% | ✅ 通过 |
| flex_attention | Enflame S60 | 0.2402 | 0.2819 | **0.85x** | 0.85 0.85 0.86 | 0.5% | ✅ 通过 |
| flex_attention | Hygon BW1000 | 0.1606 | 0.1526 | **1.06x** | 1.05 1.06 1.06 | 1.0% | ✅ 通过 |
| flex_attention | MTT S4000 | — | — | — | — | — | ❌ 编译失败 |

v0 本身只有 145µs，而本仓库所有 v1 实测的耗时下界约 110µs（见根目录各算子
results 的横向对比），所以本题的加速比上限只有约 1.3x —— 1.06x 已接近该上限，
瓶颈不在 kernel 算法。`bench/check_spill.py flex_attention` 三个 num_warps
候选均 `n_spills=0`（n_regs 171/158/156，smem 49152），寄存器不是瓶颈。

昇腾数据取 3 轮**交替**执行的中位数（各轮 1.69 1.70 1.78，极差 5.0%），明细见 `results/npu-Ascend910B2C/`。

昇腾上 v1 与沐曦走**不同分支**：预计算因果掩码（`MASK_FROM_MEM`）+ K 连续载入（`K_CONTIG`）。沐曦那一支的行为未改动。
