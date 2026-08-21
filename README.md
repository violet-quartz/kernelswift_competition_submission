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

## 跨芯片加速比总览

| 赛题 | 赛题名 | 算子 | MetaX C500 | Ascend 910B2C |
|---|---|---|---:|---:|
| Task01 | GroupedTopk | `grouped_topk` | 2.75x | **1.44x** |
| Task02 | FusedMoE | `fused_moe` | 17.58x | **17.35x** |
| Task03 | FlexAttention | `flex_attention` | 1.06x | **1.70x** |
| Task04 | SPLADESparsePooler | `SPLADE_sparse_pooler` | 2.51x | **2.04x** |
| Task05 | MusicFlamingoRotaryEmbedding | `music_flamingo_rotary_embedding` | 1.98x | **1.43x** |
| Task06 | MMEncoderAttention | `mm_encoder_attention` | 1.05x | **1.44x** |
| Task07 | mhc_post | `mhc_post` | 12.36x | **2.85x** |
| Task08 | hc_split_sinkhorn | `hc_split_sinkhorn` | 14.75x | **7.86x** |
| Task09 | CentreRandomAugmentation | `centre_random_augmentation` | 5.86x | **2.04x** |
| Task10 | head_compute_mix_bwd | `head_compute_mix_bwd` | 1.58x | 0.86x |
| | | **几何平均** | **3.68x** | **2.42x** |

两块卡都是 10/10 通过。昇腾数据取 3 轮**交替**执行的中位数，明细在各题的
`results/npu-Ascend910B2C/`；沐曦的在 `results/cuda-MetaX_C500/`。

四道题的 v1 在两块卡上走**不同的 `tl.constexpr` 分支**（`grouped_topk`、
`flex_attention`、`mm_encoder_attention`、`head_compute_mix_bwd`）。开关按**能力**
命名而不是芯片名，编译期折叠、实测运行时开销为零。分支的由来和各自的实测依据
写在对应 v1 文件的 `[KS-PORT]` 注释里。

两题的排序在两块卡上**反了**：`flex_attention` 和 `mm_encoder_attention` 在沐曦上
只有 1.06x / 1.05x，在昇腾上却是 1.70x / 1.44x。昇腾那边用的优化（因果掩码预算好
从显存载入）未必是昇腾特有的，很可能在沐曦上同样有效 —— 待复测。

`head_compute_mix_bwd` 是唯一在昇腾上慢于 v0 的一题（0.86x），原因是结构性的，
见该题 README。
