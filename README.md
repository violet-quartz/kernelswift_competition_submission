# KernelSwift 算子创新大赛

赛题与算子文件夹的对应关系。每个文件夹内有各自的 README，记录该题的实现说明和测试结果。

| 赛题 | 赛题名 | 算子文件夹 |
|---|---|---|
| Task01 | GroupedTopk | [`grouped_topk/`](grouped_topk/) |
| Task02 | FusedMoE | [`fused_moe/`](fused_moe/) |
| Task03 | FlexAttention | [`flex_attention/`](flex_attention/) |
| Task04 | SPLADESparsePooler | [`SPLADE_sparse_pooler/`](SPLADE_sparse_pooler/) |
| Task05 | MusicFlamingoRotaryEmbedding | [`music_flamingo_rotary_embedding/`](music_flamingo_rotary_embedding/) |
| Task06 | MMEncoderAttention | [`mm_encoder_attention/`](mm_encoder_attention/) |
| Task07 | mhc_post | [`mhc_post/`](mhc_post/) |
| Task08 | hc_split_sinkhorn | [`hc_split_sinkhorn/`](hc_split_sinkhorn/) |
| Task09 | CentreRandomAugmentation | [`centre_random_augmentation/`](centre_random_augmentation/) |
| Task10 | head_compute_mix_bwd | [`head_compute_mix_bwd/`](head_compute_mix_bwd/) |

`bench/`、`env/`、`run.sh` 是所有算子共用的基础设施，不属于任何单独一题。


📋 **[跨芯片适配情况 → CROSS_CHIP.md](CROSS_CHIP.md)** —— 六款国产芯片的完整实测矩阵、
测量方法，以及摩尔线程侧三个厂商工具链缺陷的最小复现（规约静默算错、`tl.dot` 编译失败、
评测脚本不识别 musa 设备）。

## 跨芯片加速比总览

| 赛题 | 赛题名 | 算子 | MetaX C500 | Ascend 910B2C | Iluvatar BI-V150 | Enflame S60 | Hygon BW1000 | MTT S4000 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Task01 | GroupedTopk | `grouped_topk` | **3.06x** | **1.46x** | **2.79x** | **1.55x** | **2.24x** | ✗ |
| Task02 | FusedMoE | `fused_moe` | **17.16x** | **17.30x** | **9.24x** | **17.75x** | **14.00x** | ✗ |
| Task03 | FlexAttention | `flex_attention` | **1.06x** | **1.70x** | 0.62x | 0.85x | **1.06x** | ✗ |
| Task04 | SPLADESparsePooler | `SPLADE_sparse_pooler` | **2.50x** | **2.08x** | **1.73x** | **1.21x** | **1.69x** | ✗ |
| Task05 | MusicFlamingoRotaryEmbedding | `music_flamingo_rotary_embedding` | **2.10x** | **1.39x** | **2.20x** | **1.78x** | **2.07x** | **3.34x** |
| Task06 | MMEncoderAttention | `mm_encoder_attention` | **1.09x** | **1.46x** | 0.63x | 0.74x | **1.06x** | ✗ |
| Task07 | mhc_post | `mhc_post` | **12.42x** | **2.88x** | **16.95x** | 0.80x | **7.45x** | **10.18x** |
| Task08 | hc_split_sinkhorn | `hc_split_sinkhorn` | **14.19x** | **7.81x** | **8.35x** | **5.11x** | **10.01x** | **26.24x** |
| Task09 | CentreRandomAugmentation | `centre_random_augmentation` | **5.70x** | **2.16x** | **3.90x** | **2.05x** | **4.19x** | ✗ |
| Task10 | head_compute_mix_bwd | `head_compute_mix_bwd` | **1.85x** | 0.85x | **1.94x** | **1.04x** | **1.03x** | ✗ |
| | | **几何平均** | **3.78x** | **2.44x** | **2.89x** | **1.78x** | **2.88x** | —（3/10） |

五块卡**全部 10/10 通过**。加粗 = 快于 v0。数据取 3 轮**交替**执行的中位数，
逐轮值和极差见各题 README 与 `results/<芯片>/`。

**沐曦那一列是早期单次测量**，没有逐轮数据，和其余四列不完全可比 ——
那台机器目前离线，等恢复后按同样方式重测。

四道题的 v1 在不同卡上走**不同的 `tl.constexpr` 分支**（`grouped_topk`、
`flex_attention`、`mm_encoder_attention`、`head_compute_mix_bwd`），另有两道
按后端选常量（`mhc_post`、`SPLADE_sparse_pooler` 的 `num_warps`）。开关按**能力**
命名而不是芯片名，编译期折叠、实测运行时开销为零。由来和实测依据写在对应 v1 文件的
`[KS-PORT]` 注释里。

**海光 BW1000 是唯一十题全部快于 v0 的卡。** 两道 attention 题在其余卡上普遍吃亏，
根源是 Inductor 会把 v0 直接降成厂商的 `_scaled_dot_product_flash_attention`，
手写 Triton 的对手其实是厂商优化过的库；昇腾例外，那里的收益来自预计算因果掩码。
