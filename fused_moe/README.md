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
│   └── ascend-910b2c/                昇腾环境配置
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

| Task | 芯片 | v0 (ms) | v1 (ms) | Speedup | 各轮实测 | 极差 | 结论 |
|---|---|---:|---:|---:|---|---:|:---:|
| fused_moe | MetaX C500 | 3.0206 | 0.1718 | **17.58x** | — | — | ✅ 通过 |
| fused_moe | Ascend 910B2C | 2.2535 | 0.1303 | **17.30x** | 17.28 17.30 19.56 | 13.2% | ✅ 通过 |
| fused_moe | Iluvatar BI-V150 | 3.0556 | 0.3305 | **9.24x** | 9.23 9.24 9.32 | 1.0% | ✅ 通过 |
| fused_moe | Enflame S60 | 5.3052 | 0.2930 | **17.75x** | 16.94 17.75 18.61 | 9.4% | ✅ 通过 |
| fused_moe | Hygon BW1000 | 3.5957 | 0.2575 | **14.00x** | 13.73 14.00 14.64 | 6.5% | ✅ 通过 |

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

⚠ **本节的 launcher 数字已修正。** 早先这里写的是「每多一次 launch 23.9 µs、
autotune 包装层 15.3 µs」，据此把「摘掉 autotune」排在第一位。那组数是异常值：
`bench/profile_overhead.py` 的 ①②②b②c 四行测的是**与 task 无关**的空 kernel，
在同一台 C500 上跑 SPLADE_sparse_pooler 两次都得到

    首次 launch 净开销   ~48 µs   （与本题那次一致）
    每多一次 launch       3.7 µs   ← 不是 23.9
    autotune 包装层       4.7 µs   ← 不是 15.3

空 kernel 的第二次 launch 只该付一遍 Python 启动路径，小的那组才符合物理。

**修正后的结论：下面第 1、2 条都不值得做。** 摘掉 autotune 只省 ~9 µs（两个 kernel
× 4.7），却要绑死芯片、换机器就得重测；合并两个 kernel 只省 ~4 µs，而规约必须
等所有专家算完，没有干净解法。剩下的空间在 kernel 本体（完整 forward 155.8 µs 里
约 74.5 µs 是 kernel + 其余 python，固定开销约 62 µs），也就是下面第 3 条那类
减少实际访存/算力的改动。

1. ~~**摘掉 autotune、把赢家写死**~~ —— 收益 ~9 µs，且赢家在两次运行之间会变
   （一次 `BLOCK_T=16, num_warps=2`，一次 `BLOCK_T=32, num_warps=4`，见
   `results.json` 的 `raw_output`）。这与逐 config 计时看到的「全表只差 1.79x」
   一致：config 差距太小，噪声就能决定名次。收益小、风险大，搁置。
2. ~~**合并两个 kernel**~~ —— 收益 ~4 µs，不值得。
3. **`partial[E,T,H]` → `partial[TOP_K,T,H]`**：8 片里只有 2 片非零。改按**排名**
   而不是专家编号索引，每个槽位仍恰好被一个 program 写一次，traffic 降 4 倍。

昇腾数据取 3 轮**交替**执行的中位数（各轮 17.12 17.35 17.79，极差 3.9%），明细见 `results/npu-Ascend910B2C/`。
