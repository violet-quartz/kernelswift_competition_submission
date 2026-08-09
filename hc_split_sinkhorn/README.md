# KernelSwift 算子创新大赛 — hc_split_sinkhorn（沐曦 C500）

用 Triton 重写 `hc_split_sinkhorn`，在沐曦 C500 上相对赛题给出的 torch 参考实现
取得 **14.30x** 加速，正确性通过官方 `auto_bench.py` 校验。

| Task | 芯片 | v0 (ms) | v1 (ms) | Speedup | 正确性 |
|---|---|---:|---:|---:|:---:|
| hc_split_sinkhorn | MetaX C500 | 1.5571 | 0.1089 | **14.30x** | PASS |

完整测量条件、稳定性说明和耗时构成见 [results/cuda-MetaX_C500/RESULTS.md](results/cuda-MetaX_C500/RESULTS.md)。

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
│   ├── count_ops.py              统计 v0 每次 forward 的 aten 算子数
│   ├── profile_overhead.py       拆解 v1 每次 forward 的固定开销构成
│   └── tasks.json                题目清单
├── env/
│   ├── capture.sh                环境快照
│   ├── selftest.py               后端连通性 + Triton 工具链自检
│   └── metax-c500/               沐曦环境配置
├── docs/
│   ├── debug-hc_split_sinkhorn.md   调试记录（编译器 bug 定位与解决全过程）
│   └── probes/                      支撑上述记录的探针脚本和原始输出
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

### 判断：这是 launch-bound，不是 compute-bound

`auto_bench.py` 的计时方式（L429-445）是**每次 forward 单独同步计时**：

```python
for _ in range(repeat):
    start = time.perf_counter()
    model.forward(*inputs)
    sync_devices()                 # 每次都同步
    samples.append(...)            # 取 median
```

而这题的数据极小：`comb` 只有 `[16,4,4]` = 256 个 float，`pre`/`post` 各 `[16,4]`。
`bench/count_ops.py` 用 `TorchDispatchMode` 实测出 v0 每次 forward 派发
**136 个 aten 算子**（20 轮 Sinkhorn，每轮 2 次除法 + 2 次求和，加上前面的
sigmoid / exp / softmax）。

在这个规模下，耗时正比于 kernel launch 次数而非 FLOPs——实测每个 aten 算子的
派发成本约 11.8 µs，而真正的计算只要 23 µs。所以优化目标是**少启动几次**，
不是算得更快。

**136 个算子 → 1 个 kernel。**

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

完整的定位过程、12 个二分用例的结果、4 个变体的交叉验证数据，见
**[docs/debug-hc_split_sinkhorn.md](docs/debug-hc_split_sinkhorn.md)**。

## 关于反作弊

赛制禁止「代码实际执行路径未运行自定义算子、全程仅使用 PyTorch 内置算子」的提交。
本作品的执行路径可以逐点核验：

- `v1/hc_split_sinkhorn.py` 的 `ModelNew.forward()` 中**没有任何 try/except、
  没有任何条件分支**，唯一的计算路径就是调用 `_hc_split_sinkhorn_kernel`
  这个 `@triton.jit` kernel。forward 里除了 3 次 `torch.empty`（分配输出）
  和 3 次 `view`（改元信息）之外，不调用任何 PyTorch 计算算子。
- 文件顶部的 `KS_IMPL = "triton"` 标记 + `bench/run_all.py` 的
  `uses_triton()` 静态检查（独立验证该文件确实 import 了 triton）双重确认。
- `bench/auto_bench.py` 与官方上游**逐字一致**，未打任何补丁。
  `run.sh` 只负责拉起子进程和汇总，不介入任何计时或比对逻辑。

## 环境说明

实测环境见 [results/cuda-MetaX_C500/RESULTS.md](results/cuda-MetaX_C500/RESULTS.md)
和 `env/metax-c500/env.lock.txt`（由 `env/capture.sh` 在机器上生成）。

`env/metax-c500/requirements.txt` 刻意**不 pin** torch 和 triton：两者都由沐曦
官方 MACA 镜像提供，版本与 MACA toolkit / 驱动强绑定，用 pip 装 PyPI 上的通用版
会覆盖掉 MACA 版，直接导致 `torch.cuda` 不可用。

### 一处可移植性处理

`get_inputs()` 返回 **CPU 张量**而非硬编码 `device="cuda"`。原因是
`auto_bench.py` L127 的 `_rewrite_device_for_backend()` 只把 `'npu'` 字面量重写成
目标后端，**不会**把 `'cuda'` 重写成 `'npu'`，所以硬编码 cuda 的文件在昇腾等后端上
会直接报错。返回 CPU 张量后，`_detect_target_device()` + `_move_to_device()`
会自动搬到当前加速器上（计时发生在搬运之后，不影响结果）。

`v0/hc_split_sinkhorn.py` 与赛题原文**逐字一致**（该题的 `get_inputs()` 本来就建
CPU 张量），仅删除了末尾的 `if __name__ == "__main__"` 自测块——`auto_bench.py`
L74 的 `_filter_module_ast()` 本来也会丢弃它。计算逻辑未改动。
