# 性能测试结果 — 沐曦 C500

## 环境

| 项 | 值 |
|---|---|
| 芯片 | MetaX C500（25% 算力切片 + 16GB 显存配额，`mx-smi` 报告） |
| 驱动 / MACA | Kernel Mode Driver 3.8.30 / MACA 3.3.0.15 |
| PyTorch | 2.8.0+metax3.3.0.2 |
| Triton | 3.0.0（沐曦分支，`triton.backends == ['metax']`） |
| Python | 3.10.10 |
| 平台 | Linux-5.15.0-58-generic-x86_64-with-glibc2.39 |

## 结果

测量由官方 `bench/auto_bench.py` 完成（与上游逐字一致，未做任何修改），
默认参数 `warmup=200 repeat=500 atol=1e-2 rtol=1e-2 seed=42`，取 500 次的中位数。

| Task | v0 (ms) | v1 (ms) | Speedup | 正确性 |
|---|---:|---:|---:|:---:|
| hc_split_sinkhorn | 1.5571 | 0.1089 | **14.30x** | PASS |

复现命令：

```bash
bash run.sh --only hc_split_sinkhorn
```

## 关于测量稳定性

同一份代码测了两次：

| 时间 (UTC) | v0 (ms) | v1 (ms) | Speedup |
|---|---:|---:|---:|
| 2026-08-08 06:04 | 1.6492 | 0.1088 | 15.16x |
| 2026-08-08 07:13 | 1.5571 | 0.1089 | 14.30x |

**v1 两次相差 0.1%，v0 相差 6%。** 抖动全在 v0 侧——它派发 136 个 aten 算子，
比只有 1 次 launch 的 v1 更容易受同物理卡上其它租户的干扰（本次测试用的是
C500 的 25% 算力切片）。

上表取两次中**较保守**的一次（14.30x）作为报告值。

## 加速比的构成

用实测数据可以把这 14.30x 拆开：

```
v0 = 136 个 aten 算子 × 约 11.8 µs/算子 ≈ 1.557 ms
v1 = 0.109 ms
     ├─ Triton kernel 本体            0.023 ms   （probe_loop_variants 实测）
     ├─ Triton Python launcher        0.048 ms
     ├─ 3 次 torch.empty              0.008 ms
     ├─ auto_bench 的 sync_devices()  0.023 ms   （计时框架自身开销，v0 也付）
     └─ 其余 python + 框架地板        0.007 ms
```

由 `bench/profile_overhead.py` 测得。这题的数据规模极小（`comb` 只有
`[16,4,4]` = 256 个 float），耗时由 **kernel launch 次数**决定而非 FLOPs，
所以优化的核心是把 136 次 launch 压成 1 次。
