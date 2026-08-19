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
    ├── scratch/                      一次性诊断脚本，不参与评测
    │   └── which_config.py           报告 autotune 选中了哪个 config + 分 kernel 计时
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

| Task | 芯片 | v0 (ms) | v1 (ms) | Speedup | 结论 |
|---|---|---:|---:|---:|:---:|
| fused_moe | MetaX C500 | 3.0206 | 0.1718 | **17.58x** | ✅ 通过 |

由 `bash run.sh --only fused_moe` 产出，原始数据见
`results/cuda-MetaX_C500/{RESULTS.md,results.json}`（2026-08-19）。

### 实现要点

`grid = (E, cdiv(T, BLOCK_T))`，每个 program 只管**一个专家 × 一批 token**，
是三次 load、三次 `tl.dot`、一次 store 的直线代码。跨专家规约不用 atomic ——
每个 program 写进 `partial[E, T, H]` fp32 里自己独占的一片，再由第二个 kernel
沿 E 求和并转 fp16。权重在首次 forward 时一次性预转置 + 预降精度并缓存，
kernel 里零 `tl.trans`、零 dtype 转换。

四块设计决策记在 `v1/fused_moe.py` 的注释里：`[KS-SHAPE]`（并行形状，含被实测
否掉的那个形状及其数据）、`[KS-SMEM]`（shared memory 预算怎么定 BLOCK_T 上限）、
`[KS-TUNE]`（三个旋钮为什么不写死）、`[KS-CACHE]`（权重预处理与缓存失效判据）。

### 诊断工具

```bash
python3 bench/check_spill.py fused_moe          # n_regs / n_spills
python3 bench/profile_overhead.py fused_moe     # 拆固定开销（launcher / autotune / 分配）
python3 fused_moe/scratch/which_config.py       # autotune 选了哪个 config + 分 kernel 计时
```

### 已知的、尚未做的优化

按实测收益排序（数据来自 `bench/profile_overhead.py`，完整 forward 155.8 µs）：

1. **摘掉 autotune、把赢家写死**：Autotuner 包装层每次调用约 15.3 µs，两个 kernel
   共 ~30 µs（20%）。

   ⚠ 但**赢家在两次运行之间会变**：同一台 C500 上，一次选出
   `_moe_expert_kernel: BLOCK_T=16, num_warps=2`，另一次选出 `BLOCK_T=32, num_warps=4`
   （见 `results.json` 的 `raw_output` 与 `scratch/which_config.py` 的输出）。
   这与逐 config 计时看到的「全表只差 1.79x」一致 —— config 之间差距太小，
   噪声就能决定名次。所以写死之前得先搞清楚到底哪个更快、还是根本无所谓；
   直接各写死一个实测比较，比信 autotune 的单次结论可靠。
2. **合并两个 kernel**：每多一次 launch 的边际成本 23.9 µs。但规约必须等所有专家
   算完，目前没有干净解法（fp16 atomic 要先清零 `out`，那又是一次 launch）。
3. **`partial[E,T,H]` → `partial[TOP_K,T,H]`**：8 片里只有 2 片非零。改按**排名**
   而不是专家编号索引，每个槽位仍恰好被一个 program 写一次，traffic 降 4 倍。