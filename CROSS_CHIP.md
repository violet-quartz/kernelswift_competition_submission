# 跨芯片适配情况

本仓库的 v1 实现在 **6 款国产 AI 芯片**上实测过。同一份代码通过
`tl.constexpr` 分支适配不同后端，开关按**能力**命名而非芯片名，编译期折叠、
运行时开销实测为零。

## 总览

| 芯片 | 通过 | 几何平均 | 备注 |
|---|---|---|---|
| 沐曦 MetaX C500 | **10/10** | **3.78x** | 六卡中最高 |
| 天数智芯 Iluvatar BI-V150 | **10/10** | **2.89x** | 噪声最低（极差中位 1.2%），最适合做配对比较 |
| 海光 Hygon BW1000 | **10/10** | **2.88x** | **唯一十题全部快于 v0 的卡** |
| 昇腾 Ascend 910B2C | **10/10** | **2.44x** | |
| 燧原 Enflame S60 | **10/10** | **1.78x** | 噪声最高（极差中位 11%、最大 42.9%） |
| 摩尔线程 MTT S4000 | 3/10 | — | 厂商工具链缺陷，见下方专节 |

逐题数据见各算子目录下的 `results/<芯片>/`，含 3 轮交替执行的逐轮值与极差。

## 各题 × 各芯片加速比

| 赛题 | 算子 | MetaX C500 | Ascend 910B2C | Iluvatar BI-V150 | Enflame S60 | Hygon BW1000 | MTT S4000 |
|---|---|---:|---:|---:|---:|---:|---:|
| Task01 | `grouped_topk` | 3.06x | 1.46x | 2.79x | 1.55x | 2.24x | ✗ |
| Task02 | `fused_moe` | 17.16x | 17.30x | 9.24x | 17.75x | 14.00x | ✗ |
| Task03 | `flex_attention` | 1.06x | 1.70x | 0.62x | 0.85x | 1.06x | ✗ |
| Task04 | `SPLADE_sparse_pooler` | 2.50x | 2.08x | 1.73x | 1.21x | 1.69x | ✗ |
| Task05 | `music_flamingo_rotary_embedding` | 2.10x | 1.39x | 2.20x | 1.78x | 2.07x | **3.34x** |
| Task06 | `mm_encoder_attention` | 1.09x | 1.46x | 0.63x | 0.74x | 1.06x | ✗ |
| Task07 | `mhc_post` | 12.42x | 2.88x | 16.95x | 0.80x | 7.45x | **10.18x** |
| Task08 | `hc_split_sinkhorn` | 14.19x | 7.81x | 8.35x | 5.11x | 10.01x | **26.24x** |
| Task09 | `centre_random_augmentation` | 5.70x | 2.16x | 3.90x | 2.05x | 4.19x | ✗ |
| Task10 | `head_compute_mix_bwd` | 1.85x | 0.85x | 1.94x | 1.04x | 1.03x | ✗ |

摩尔线程上通过的三题里，`hc_split_sinkhorn` 的 **26.24x 是全部六卡中的最高单题加速比**。

## 摩尔线程 MTT S4000：三个厂商工具链缺陷

**这三条都不是本仓库代码能规避的**，全部有最小复现。测试环境：
`triton-musa 3.6.0`（`backends=['musa']`，装自官方源
`https://dl.mthreads.com/repo/api/pypi/pypi/simple`）+ `torch_musa 1.3.0` + `torch 2.2.0`。

### 1. N>32 的规约**静默算错**（最严重）

编译通过、程序正常跑完、无任何报错，但结果是错的：

| N | `tl.sum` 实得 | 正确值 | `tl.max` 实得 | 正确值 |
|---|---|---|---|---|
| 32 | 496 | 496 ✅ | 31 | 31 ✅ |
| 64 | 992 | 2016 ❌ | 31 | 63 ❌ |
| 128 | 1984 | 8128 ❌ | 31 | 127 ❌ |
| 512 | 32512 | 130816 ❌ | 127 | 511 ❌ |
| 1024 | 327168 | 523776 ❌ | 639 | 1023 ❌ |

只有 N≤32（单 warp 宽度）正确。`max` 的返回值暴露了机理：**只规约了一部分数据就返回**。

复现（`num_warps=1`）：
```python
@triton.jit
def k(x, o, N: tl.constexpr):
    tl.store(o, tl.sum(tl.load(x + tl.arange(0, N)), axis=0))
x = torch.arange(128, device="musa", dtype=torch.float32)
k[(1,)](x, o, N=128, num_warps=1)   # → 1984，正确值 8128
```

⚠ 幸而 `auto_bench` 有 v0/v1 数值对拍，才把这类结果拦下来。

### 2. `tl.dot`（矩阵乘）编译失败，与 warp 数无关

```
LLVM ERROR: Cannot select: load<... from got, align 8>
Running pass 'MTGPU DAG->DAG Pattern Instruction Selection'
```

同为 `num_warps=1` 的对照测试：

| 操作 | 结果 |
|---|---|
| 2D load + store | ✅ |
| `tl.trans` | ✅ |
| 2D→1D 规约 `tl.max(axis=1)` | ✅ |
| **`tl.dot`** | **❌ 崩溃** |

依赖 `tl.dot` 的三题（`fused_moe`、`flex_attention`、`mm_encoder_attention`）因此全部编译失败。

### 3. `num_warps>1` 时跨 warp 规约撞 `@global_smem` 的 GOT load

Triton 用 `@global_smem`（`external addrspace(3)` 符号）做跨 warp 中转，
取其地址需走 GOT，而 MTGPU 后端不会 select 这个 GOT load。

`mhc_post` 走 `num_warps=1` 从编译失败变为 10.18x —— **但这个绕法只对该题成立**，
因为它的 kernel 主体是外积累加、全程无规约。对依赖规约的题，这个绕法只会把
"编译失败"变成上述第 1 条的"静默算错"，更危险，故未采用。

### 4. 评测脚本 `auto_bench.py` 不识别 musa 设备

`_iter_accelerators()` 只遍历 `(gcu, cuda, npu, mlu)`，摩尔线程走的
`torch.musa` 不在其中。实测：`torch.musa.is_available()` 为 True、Triton 也能跑，
但 `_detect_target_device()` 直接抛 `no accelerator device available`，
**任何提交都无法运行**；同时 `sync_devices()` 会静默变成空操作。

本仓库这一侧已做对（`_ks_bootstrap` 补上了 `torch_musa` 导入 —— 不显式导入的话
`torch.musa` 压根不存在）。评测脚本侧需在该元组中加入 `"musa"`。
本仓库的 `bench/auto_bench.py` 保持与上游逐字一致，未做修改。

### 相关公开记录

摩尔线程官方 issue [torch_musa#147](https://github.com/MooreThreads/torch_musa/issues/147)
报告了 torch_musa **算子层**在超大张量（~3e9 元素）上的静默错误结果，至今无官方回复。
与本文第 1 条不是同一个 bug（那是算子层、这是 Triton 编译器层；那是 3e9 元素、
这是 32 个元素），但说明该软件栈在多个层面存在同类问题。

Triton-MUSA **没有公开代码仓库**（摩尔线程 GitHub 有 128 个仓库、含多个 `-musa`
移植，但无 triton-musa），仅以二进制 wheel 发布，故上述缺陷无公开 issue 可追踪。

## 沐曦 C500 说明

该机器在中途出现过驱动层故障（`mxc_queue_acquire failed`，一个 4×4 矩阵乘即可
复现，而 `mx-smi` 仍报卡空闲可用），期间无法验证。**故障排除后已用最终代码
完整复测**，数据与其余各卡同规格（3 轮交替执行取中位数）。

复测结果与故障前的历史单次测量一致，无回归；几处提升（`head_compute_mix_bwd`
1.58x→1.85x、`grouped_topk` 2.75x→3.06x）都落在该卡的噪声带内 —— 沐曦极差
中位 8.8%、最大 43.5%，是六卡中第二抖的，仅次于燧原。

## 测量方法

- **3 轮交替执行**：同批各 task 轮流跑，而非同一 task 连跑 N 次 —— 让同批 job
  共享同一份环境漂移，差值里的共模噪声才能抵消。
- 表中取各轮**中位数**；逐轮值与极差记录在 `results/<芯片>/results.json`。
- 各卡噪声水平差异很大（天数 1.2% / 昇腾 5% / 沐曦 8.8% / 燧原 11%），比较改动优劣时
  必须先看该卡的噪声带 —— 燧原上 15% 以内的差异不具备统计意义。
