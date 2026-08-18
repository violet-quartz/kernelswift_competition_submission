# KernelSwift 算子创新大赛 — fused_moe

## 目录结构

`bench/`、`env/`、`run.sh` 是**所有算子共用**的基础设施，放在仓库根目录，
不属于本文件夹；每个算子一个文件夹（本文件夹就是 `fused_moe` 这一题），
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
└── fused_moe/                        本文件夹
    ├── README.md                     本文件
    ├── tasks/
    │   └── fused_moe.py              赛题原始文件（未修改，供对照）
    ├── v0/fused_moe.py               torch 参考实现（Model）—— 加速比的基准
    ├── v1/fused_moe.py               Triton 优化实现（ModelNew）—— 参赛作品
    ├── spill_probe.py                本题的 spill 探针，供 bench/check_spill.py 调用
    └── results/                      性能测试结果
```

`spill_probe.py` 提供模块级 `warmup(dev)`，只编译不执行 v1 的 kernel，让
`bench/check_spill.py` 读出 `n_regs / n_spills`。本题的 kernel 挂了
`@triton.autotune`，探针会按「每线程寄存器数 ∝ (12288 + 456·BLOCK_T) / num_warps」
挑出最容易溢出的那个 config 来编译；另有 `warmup_all(dev)` 可把全部 config
挨个编一遍对照（通用驱动不调，手动用）。

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

# 2. 跑对拍 + 计时（--only 按 task 名过滤，本算子叫 fused_moe）
bash run.sh --only fused_moe
```

`run.sh` 会为每个 task 拉起一个独立的 `auto_bench.py` 子进程，
结果写入本文件夹下的 `results/<chip>/{RESULTS.md,results.json}`。


## 环境说明

实测环境见 `results/cuda-MetaX_C500/RESULTS.md`
和 `env/metax-c500/env.lock.txt`（由 `env/capture.sh` 在机器上生成）。

## 测试结果

v1 已实现，尚未上机，暂无成绩。

思路：单 kernel 融合路由 + dispatch + top_k 规约，grid 按 token 分片
（`grid = (cdiv(T, BLOCK_T),)`），每个 program 内部循环全部 8 个专家、用稠密的
`gate_w[BLOCK_T, E]` 把未选中的专家权重置 0 —— 于是 v0 里那 8 次 host 同步和
16 次布尔索引 gather/scatter 全部消失，也不需要 atomic 或第二个 kernel。
权重在首次 forward 时一次性预转置 + 预降精度并缓存，kernel 里零 `tl.trans`、
零 dtype 转换。`BLOCK_T` / `num_warps` / `num_stages` 交给 `@triton.autotune`。

推导和踩坑都记在源码注释里：`v0/fused_moe.py` 顶部的 KS-PORT 是 harness 契约，
`v1/fused_moe.py` 里的 KS-SHAPE（并行形状）、KS-CACHE（权重缓存）、
KS-TUNE（三个旋钮为什么不写死）分别对应三处设计决策。

| Task | 芯片 | v0 (ms) | v1 (ms) | Speedup | 结论 |
|---|---|---:|---:|---:|:---:|
| fused_moe | MetaX C500 | 3.1371 | 0.7386 | **4.25x** | ✅ 通过 |

实测中有两条推翻了设计假设，记在 `v1/fused_moe.py` 的 `[KS-MEASURED]` 里：
本题**不是 launch-bound**（kernel 676 us，host + launch 只有 3.8 us），
而且六个 autotune config 的差距只有 1.79x —— 旋钮已经调到头，还剩的空间在结构上。
`fused_moe/scratch/which_config.py` 可复现这两组数据。
