# KernelSwift 算子创新大赛 — mhc_post

用 Triton 重写 `mhc_post`，正确性通过官方 `auto_bench.py` 校验。

以下是在沐曦 C500 上相对赛题给出的 torch 参考实现取得的加速：

| Task | 芯片 | v0 (ms) | v1 (ms) | Speedup | 正确性 |
|---|---|---:|---:|---:|:---:|
| mhc_post | MetaX C500 | 6.1411 | 0.4923 | **12.47x** | PASS |

完整测量条件见 [results/cuda-MetaX_C500/RESULTS.md](results/cuda-MetaX_C500/RESULTS.md)。

## 目录结构

`bench/`、`env/`、`run.sh` 是**所有算子共用**的基础设施，放在仓库根目录，
不属于本文件夹；每个算子一个文件夹（本文件夹就是 `mhc_post` 这一题），内部
只放这道题自己的东西。仓库里如果还有别的算子文件夹，这里不重复画出——结构
跟本文件夹是同一个模式，各自有自己的 README：

```
仓库根目录/
├── run.sh                            一键运行脚本（对拍 + 计时 + 产出结果，所有算子共用）
├── bench/
│   ├── auto_bench.py                 官方评测脚本（与上游逐字一致，未做任何修改）
│   ├── run_all.py                    批量拉起 auto_bench + 汇总结果，按 tasks.json 里的 name 逐个跑
│   ├── check_spill.py                通用寄存器溢出诊断驱动，按算子文件夹名找 spill_probe.py
│   └── tasks.json                    所有算子共用的 task 清单，只有 name，运行时按 name 找同名算子文件夹
├── env/
│   ├── capture.sh                    环境快照
│   ├── selftest.py                   后端连通性 + Triton 工具链自检，自动发现每个算子的 selftest_probe.py
│   ├── metax-c500/                   沐曦环境配置
│   └── ascend-910b3/                 昇腾环境配置
└── mhc_post/                         本文件夹
    ├── README.md                     本文件
    ├── spill_probe.py                 寄存器溢出诊断探针，供 bench/check_spill.py 通用驱动调用
    ├── tasks/
    │   └── mhc_post.py                赛题原始文件（未修改，供对照）
    ├── v0/mhc_post.py                 torch 参考实现（Model）—— 加速比的基准
    ├── v1/mhc_post.py                 Triton 优化实现（ModelNew）—— 参赛作品
    ├── scratch/                       torch.compile 对照实验（Inductor 生成代码 dump，判断编译器融合程度用）
    └── results/                       性能测试结果
```

`bench/run_all.py` 读根目录共享的 `bench/tasks.json` 决定跑哪些 task，
每个 task 的 `name` 就是仓库根目录下同名的算子文件夹（跟 `bench/check_spill.py`
认 task 名的方式一致）。`env/selftest.py` 的探针发现是另一套机制——按"同时有
`v0/` 和 `v1/` 子目录"自动扫描，不需要在 `tasks.json` 里登记。

## 快速开始

**以下命令都在仓库根目录下执行**（`bench/`、`env/`、`run.sh` 都在那里，不在本文件夹里）：

```bash
# 1. 环境准备（首次上机）
# 以沐曦为例，若为昇腾，则是 bash env/ascend-910b3/setup.sh
bash env/metax-c500/setup.sh

# 2. 跑对拍 + 计时（--only 按 task 名过滤，本算子叫 mhc_post）
bash run.sh --only mhc_post
```

`run.sh` 会为每个 task 拉起一个独立的 `auto_bench.py` 子进程，
结果写入本文件夹下的 `results/<chip>/{RESULTS.md,results.json}`。

## 优化思路

### 判断：这是 memory-bound，不是 compute-bound

`mhc_post` 的计算是一次 `einsum('abmn,abmc->abnc', comb_res_mix, residual.float())`
加一次广播乘法加法。`einsum` 长得像矩阵乘——1.2 的经验法则里 matmul 该是
compute-bound——但形状拆开看会发现它是伪装的：`comb_res_mix` 是
`(n0,n1,4,4)`，`residual` 是 `(n0,n1,4,1280)`，对每个位置是一次
`(4×4)^T @ (4×1280)`，**归约维度 K 只有 4**。真正的 compute-bound matmul
靠数据在 shared memory/寄存器里反复复用来喂饱 tensor core，K=4 太小，几乎
没有复用空间，行为上更接近逐元素算子。

粗算一下算术强度：

| | 数值 |
|---|---|
| 输入字节 | x(bf16,20MB) + residual(bf16,80MB) + post_layer_mix(fp32,0.13MB) + comb_res_mix(fp32,0.5MB) |
| 输出字节 | output(bf16) ≈ 80MB |
| 理论最小 HBM 流量 | ≈ 180MB（读一次输入 + 写一次输出） |
| FLOPs | einsum(2×M×N×K，M=N=4,K=4,batch=8192) + 广播乘 + 加法 ≈ 0.42 GFLOP |
| 算术强度 | 420M / 180M | ≈ 2.3 FLOP/byte |

2.3 FLOP/byte 远低于 fp32 类算子十几到大几十 FLOP/byte 的经验 ridge point，所以可以采取 Fusion 的方式进行优化。

### 为什么手写能打得过 `torch.compile`：extern bmm 是硬融合边界

上机前先用 `scratch/probe_inductor.py` 跑出 `torch.compile` 的编译产物
（`scratch/output_code.py`），逐个 kernel 数流量：Inductor 确实把尾部的
`x.float()` + 广播乘 + 加法 + `.bfloat16()` 融进了一个 kernel，总流量从
v0 朴素 eager 执行的 ≈1.54GB 打到了 ≈821MB。但 `einsum` 被下发成
`extern_kernels.bmm(...)`——调用 cuBLAS 做批量矩乘。**extern kernel 调用是
编译器过不去的硬融合边界**：`residual.float()` 必须先单独物化成 160MB 的
fp32 buffer 才能喂给 bmm，bmm 的输出 `term2`（另外 160MB）也必须先写回显存，
下一个 kernel才能读回来——这两块加起来占了 Inductor 版本总流量的 68%。

手写直接跳过 cuBLAS：K=4 小到用不着分块矩乘的机制，在寄存器里做 4 次乘加
就够了，`residual.float()` 和 `term2` 全程不落显存。理论最小流量 180MB。

### 实现

`grid = (n0*n1,) = (8192,)`，每个 program 独占一个位置：一次性读入
`residual[a,b,:,:]`（4×1280，升精度成 fp32）、`x[a,b,:]`（1280，升精度），
`mhc_mult=4` 这一维用 `tl.static_range` 展开，循环体里对每个输出分支 `i`
直接按地址读 `comb_res_mix[a,b,:,i]`（4 个 float，见下方「关键难点」），
在寄存器里做 4 次乘加得到 `term2[i,:]`，算出 `output[i,:] = x*post_layer_mix[i]+term2[i,:]`
后立即降精度写出——`residual`/`x` 只读一次，`term2` 从不落显存。


## 环境说明

实测环境见 [results/cuda-MetaX_C500/RESULTS.md](results/cuda-MetaX_C500/RESULTS.md)
和 `env/metax-c500/env.lock.txt`（由 `env/capture.sh` 在机器上生成）。

`env/metax-c500/requirements.txt` 刻意**不 pin** torch 和 triton：两者都由沐曦
官方 MACA 镜像提供，版本与 MACA toolkit / 驱动强绑定。

### 寄存器溢出检查

`H=1280` ，`BLOCK_H` 又按 2 的幂上取整
到 2048（比实际需要的 1280 多垫了 60%），寄存器压力比那道题更值得关注。实测：

```
python3 bench/check_spill.py mhc_post
n_regs=88  n_spills=0  smem=8192
```

`n_spills=0`，`BLOCK_H` 的冗余没有把寄存器压爆，不用为了控制寄存器再拆分块。


### 一处可移植性处理

`get_inputs()` 返回 **CPU 张量**而非硬编码 `device="cuda"`。原因是
`auto_bench.py` L127 的 `_rewrite_device_for_backend()` 只把 `'npu'` 字面量重写成
目标后端，**不会**把 `'cuda'` 重写成 `'npu'`，所以硬编码 cuda 的文件在昇腾等后端上
会直接报错。返回 CPU 张量后，`_detect_target_device()` + `_move_to_device()`
会自动搬到当前加速器上（计时发生在搬运之后，不影响结果）。
