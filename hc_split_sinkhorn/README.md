# KernelSwift 算子创新大赛 — hc_split_sinkhorn

## 目录结构

`bench/`、`env/`、`run.sh` 是**所有算子共用**的基础设施，放在仓库根目录，
不属于本文件夹；每个算子一个文件夹（本文件夹就是 `hc_split_sinkhorn` 这一
题），内部只放这道题自己的东西。仓库里如果还有别的算子文件夹，这里不重复
画出——结构跟本文件夹是同一个模式，各自有自己的 README：

```
仓库根目录/
├── run.sh                            一键运行脚本（对拍 + 计时 + 产出结果，所有算子共用）
├── bench/
│   ├── auto_bench.py                 官方评测脚本（与上游逐字一致，未做任何修改）
│   ├── run_all.py                    批量拉起 auto_bench + 汇总结果，按 tasks.json 里的 name 逐个跑
│   └── tasks.json                    所有算子共用的 task 清单，只有 name，运行时按 name 找同名算子文件夹
├── env/
│   ├── capture.sh                    环境快照
│   ├── selftest.py                   后端连通性 + Triton 工具链自检，同样自动发现每个算子的 selftest_probe.py
│   ├── metax-c500/                   沐曦环境配置
│   └── ascend-910b3/                 昇腾环境配置
└── hc_split_sinkhorn/                本文件夹
    ├── README.md                     本文件
    ├── selftest_probe.py             v1 kernel 特性冒烟探针，供 env/selftest.py 自动发现
    ├── tasks/
    │   └── hc_split_sinkhorn.py      赛题原始文件（未修改，供对照）
    ├── v0/hc_split_sinkhorn.py       torch 参考实现（Model）—— 加速比的基准
    ├── v1/hc_split_sinkhorn.py       Triton 优化实现（ModelNew）—— 参赛作品
    ├── scratch/                      torch.compile 对照实验（Inductor 生成代码 dump，判断编译器融合程度用）
    └── results/                      性能测试结果
```

`bench/run_all.py` 读根目录共享的 `bench/tasks.json` 决定跑哪些 task，
每个 task 的 `name` 就是仓库根目录下同名的算子文件夹（跟 `bench/check_spill.py`
认 task 名的方式一致），新增算子只需要在 `tasks.json` 里加一行 `name`。
`env/selftest.py` 的探针发现是另一套机制——按"同时有 `v0/` 和 `v1/` 子目录"
自动扫描，不需要在 `tasks.json` 里登记（见下面「环境自检」一节）。

## 快速开始

以沐曦为例

**以下命令都在仓库根目录下执行**（`bench/`、`env/`、`run.sh` 都在那里，不在本文件夹里）：

```bash
# 1. 环境准备（首次上机）
bash env/metax-c500/setup.sh

# 2. 跑对拍 + 计时（--only 按 task 名过滤，本算子叫 hc_split_sinkhorn）
bash run.sh --only hc_split_sinkhorn
```

`run.sh` 会为每个 task 拉起一个独立的 `auto_bench.py` 子进程，
结果写入本文件夹下的 `results/<chip>/{RESULTS.md,results.json}`。


## 环境说明

实测环境见 [results/cuda-MetaX_C500/RESULTS.md](results/cuda-MetaX_C500/RESULTS.md)
和 `env/metax-c500/env.lock.txt`（由 `env/capture.sh` 在机器上生成）。

