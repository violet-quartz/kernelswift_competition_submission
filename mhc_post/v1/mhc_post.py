"""v1 Triton 优化实现 — mhc_post.py

一个 kernel 融掉 v0 的 6 个算子（3×Cast + BatchMatMul + Mul + Add），
`residual` 的 fp32 副本和 `term2` 全程不落显存。

昇腾 profiling 驱动的第二版改写
------------------------------
第一版（BLOCK_H = next_power_of_2(H) = 2048，一发覆盖整个 H）在沐曦上拿到
12.47x，但昇腾上只有 1.85x。用 bench/profile_chip.py 看 AI Core 流水线，
问题定位得很清楚（昇腾 910B3 实测，每次调用）：

    墙钟 988 us   核忙 984 us   BlockNum 8192   等待 0.1 us
      aiv_mte2_time  429.8 us  43.5%   搬入
      aiv_vec_time   338.3 us  34.2%   计算
      aiv_mte3_time  327.7 us  33.2%   搬出
      aiv_scalar_time 160.4 us 16.2%   地址计算
      流水线时间之和 1256 / 核忙 984 = 重叠系数 1.28x

三条结论：
  1. `等待≈0`、`核忙≈墙钟` —— 不是调度/launch 开销，grid=8192 没问题
  2. 没有任何一条流水线接近饱和，但重叠系数只有 1.28x —— 四条流水线在**串行**
     跑（读完→算完→写完），本该并行。若能完全重叠，墙钟下限是最慢的那条
     429.8 us，相比 988 us 有 2.3x 空间
  3. 连 MTE2 自己的效率也差：搬入约 105MB 用了 429.8 us ≈ 246 GB/s，而 v0 里
     厂商的 aclnnAdd 是 691 GB/s（且 mte2_ratio 高达 0.99）—— 差 2.8x

对应做了两处改动。BLOCK_H 做成可扫的参数之后，把两者的贡献分离开了 ——
**当初的证据强度排序完全反了**。

昇腾 910B3 上的 BLOCK_H 扫描（外积版本，v0 ≈ 2.22 ms）：

    BLOCK_H   循环次数   v1(ms)    speedup
        256        5     1.2847      1.75x
        512        3     1.0697      2.08x
       1024        2     0.8724      2.56x
       2048        1     0.8201      2.71x   <- next_power_of_2(1280)，采用

  * **BLOCK_H 该取大不取小（当初判断反了）。** 当初把它从 2048 改成 256，理由是
    "1280=5×256 整除、零 padding，且 padding 破坏了访存连续性"。**这个论证是
    错的**：被 mask 掉的 lane 不发起访存，第一版实际访问的地址正好是
    m*1280+h (h<1280)，覆盖 0..5119，本来就连续；padding 只浪费向量槽位，
    并不多搬字节。而改成 256 等于把一次 4096 字节的大块传输拆成 5 次 512 字节
    的小传输，MTE2 每次搬运有固定开销，这个粒度太小 —— 实测 aiv_mte2 从
    429.8 翻倍到 877.3 us，有效带宽 244 -> 120 GB/s；scalar 也 +76%（循环 5 次，
    地址计算重复 5 遍）。
    **教训：大块传输 + 一点 padding 浪费，好过零浪费 + 小块传输。**

  * **外积累加取代 tl.sum 规约（当初标【弱】、后来还误判成"无关紧要"，
    实际是最大的一笔收益）。** 在 BLOCK_H=2048 这个点上，循环只跑一次，结构跟
    第一版完全一样（同样块大小、同样单发覆盖），唯一差别就是这一条：

        第一版  2D tile + tl.sum ×4      1.2067 ms   1.85x
        现在    M×1D load + 外积累加      0.8201 ms   2.71x   <- 1.47x

    收益大概率不在向量计算上（vec_time 变化不大），更可能在访存模式：M 次
    独立 1D load 比一个 (M, BLOCK_H) 的 2D tile load 更适合这个后端。
    附带好处：不需要 `tile[:, i]` 那种切片+整数索引（这个后端不支持，见下），
    且 M 可以保持成运行时参数而不用写死。

  * **H 方向的循环本身（未证实有收益）。** 中间版本（BLOCK_H=256，循环 5 次）
    测到重叠系数 1.28x -> 1.76x，编译器确实做起了跨迭代流水；但最优点
    BLOCK_H=2048 循环只跑一次，等于没有跨迭代可流水。也就是说流水的收益远
    抵不过小块传输的损失。要同时拿到"大块传输"和"可流水的循环"，得把循环换到
    N 方向（每个 program 处理多个位置），那是下一步要试的。

**方法学教训**：上一轮把三处改动打包一起测，不仅归因不了，还得出了方向相反的
结论 —— 当时拿"第一版 2048+规约"跟"第二版 256+外积"比，块大小的恶化把外积的
收益整个抵消掉，于是误判成"外积无关紧要"。分步扫描才把真相还原出来。

调参用的环境变量
----------------
    KS_BLOCK_H=1024 python3 ...     # H 方向分块大小
    KS_NUM_WARPS=4  python3 ...
    KS_NUM_STAGES=3 python3 ...     # 有了 H 循环才有作用对象

都不设时行为跟没有这段代码一样（BLOCK_H 由 _pick_block_h 挑，warps/stages 用
后端默认值）。BLOCK_H=2048 时循环恰好只跑 1 次，退化成第一版的结构但用外积
—— 扫这个点能顺带隔离出"外积 vs 规约"到底有没有影响。

外积累加：怎么做到既泛化 M 又不用规约
------------------------------------
朴素写法是构造 (M, BLOCK_H) 的 residual tile，再对每个输出分支 i 做
`tl.sum(comb[:, i][:, None] * r, axis=0)` —— 沿外层维度规约。

这里换个方向，把 m 提到外层循环，累加到一个 (M, BLOCK_H) 的累加器上：

    for m in static_range(M):
        r_m   = residual[m, :]                  # 1D，每行只读一次
        c_row = comb[m, :]                      # 1D，长度 M
        acc  += c_row[:, None] * r_m[None, :]   # 外积，(M,1)×(1,BLOCK_H)

    acc[i, h] = sum_m comb[m, i] * residual[m, h]    <- 正是要的 term2

好处：residual 每行仍然只读一次（m 循环里各读一次），全程没有规约，而且
不需要 `tile[:, i]` 那种切片+整数索引（这个后端不支持，见下方「踩过的坑」）。
M 保持成运行时传入的 constexpr，不用写死。

m 循环用 tl.static_range 展开，所以 acc 在 IR 层面是纯 SSA、不是循环携带变量
—— 这点对沐曦很重要，正是它踩过的段错误触发条件之一。

踩过的坑：Triton 不支持切片+整数的混合降维索引
---------------------------------------------
第一版想用 `comb_res_mix[:, i]` 从 (M,M) tile 里取第 i 列，三次报错才找到根因：

    for i in range(M):        -> unsupported tensor index: int32[]
                                 （range 编译成真循环，i 是运行时标量）
    for i in tl.static_range(M) -> unsupported tensor index: constexpr[0]
                                 （i 是编译期常量了，但类型是 constexpr 包装）
    用 i.value 拆出裸 int      -> unsupported tensor index: 0
                                 （已经是裸 int 还是不行 —— 跟类型无关，
                                   这个后端压根不支持 tile[:, k] 这种取法）

本版从结构上就不需要这种索引：取的都是 comb 的**行**（`comb[m, :]`，扁平布局
里是连续的 M 个元素，直接按地址 1D load），不是列。

刻意不设 num_warps / num_stages
-------------------------------
沐曦上 hc_split_sinkhorn 撞过 make_ttgir 段错误，而 num_warps 直接影响那个
pass 的 layout 分配。这里先用默认值把这三处结构改动的效果测出来，调优留到后面。
"""

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _env_int(name: str):
    """读一个整数环境变量，没设或非法就返回 None。

    [KS-PORT] 为什么读环境变量必须写在函数里、不能写成模块级的
    `_BLOCK_H = int(os.environ.get(...))`：auto_bench.py L74 的
    _filter_module_ast() 只保留 Import / ClassDef / FunctionDef / **字面量**赋值，
    模块级带函数调用的赋值会被整个丢弃 —— 跟 _ks_bootstrap() 必须包成函数
    是同一个原因。这里的调用点在 ModelNew.__init__ 里，函数体内不受那个过滤器影响。
    """
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        print(f"[mhc_post] 忽略非法的 {name}={raw!r}")
        return None


def _ks_bootstrap():
    """按需导入后端扩展，让 torch.npu / torch.mlu 命名空间真正出现。

    [KS-PORT] 为什么必须有这个函数、又为什么它长这样：
      * 昇腾要 `import torch_npu`、寒武纪要 `import torch_mlu`，否则 torch 上
        压根不存在 .npu / .mlu 属性。而 auto_bench.py L206 的 _iter_accelerators()
        正是用 getattr(torch, "npu", None) 来探测设备的 —— 没导入扩展，
        它就探测不到加速器，L494 直接抛 "no accelerator device available"。
      * 沐曦 C500 走 torch.cuda 命名空间，不需要任何扩展，所以这里必须能容忍
        ImportError 而不是硬 import。
      * 那为什么不写成模块级的 try/except？因为 auto_bench.py L74 的
        _filter_module_ast() 只保留 Import / ClassDef / FunctionDef / 字面量赋值
        四类节点，模块级的 try/except 是 ast.Try，**会被整个丢弃**。
        包进函数体里才能存活 —— 函数体内部不受那个过滤器影响。
      * 调用点放在 get_init_inputs() / get_inputs() 开头，因为 auto_bench.py
        L378-409 是先调这两个函数，之后才做设备探测（L516）。
    """
    import importlib

    for _mod in ("torch_npu", "torch_mlu"):
        try:
            importlib.import_module(_mod)
        except ImportError:
            pass


@triton.jit
def _mhc_post_kernel(
    x_ptr,               # [N, H]      bfloat16
    residual_ptr,        # [N, M, H]   bfloat16
    post_layer_mix_ptr,  # [N, M, 1]   float32
    comb_res_mix_ptr,    # [N, M, M]   float32
    output_ptr,          # [N, M, H]   bfloat16  out
    H: tl.constexpr,
    M: tl.constexpr,
    BLOCK_H: tl.constexpr,   # H 方向分块，宿主侧挑能整除 H 的 2 的幂
    BLOCK_M: tl.constexpr,   # = next_power_of_2(M)，tl.arange 要求 2 的幂
):
    pid = tl.program_id(0)          # 一个 program 负责一个 (a,b) 位置
    row_x = x_ptr + pid * H
    row_res = residual_ptr + pid * M * H
    row_pm = post_layer_mix_ptr + pid * M
    row_comb = comb_res_mix_ptr + pid * M * M
    row_out = output_ptr + pid * M * H

    mm = tl.arange(0, BLOCK_M)
    m_m = mm < M                    # M 不是 2 的幂时盖住 padding 出来的行

    # post_layer_mix 跟 h 无关，在 H 循环外读一次
    pm = tl.load(row_pm + mm, mask=m_m, other=0.0)          # (BLOCK_M,)

    # 按 H 分块循环。BLOCK_H 整除 H 时 mask 恒为真、零浪费；
    # 不整除时靠 mask 兜住尾块，正确性不依赖整除。
    for h0 in range(0, H, BLOCK_H):
        hh = h0 + tl.arange(0, BLOCK_H)
        m_h = hh < H

        # 外积累加：acc[i, c] = sum_m comb[m, i] * residual[m, c]
        # m 提到外层，residual 每行只读一次，全程没有规约
        acc = tl.zeros([BLOCK_M, BLOCK_H], dtype=tl.float32)
        for m in tl.static_range(M):
            # comb_res_mix[m, :] —— 扁平布局里是连续的 M 个元素，1D load
            c_row = tl.load(row_comb + m * M + mm, mask=m_m, other=0.0)
            r_m = tl.load(row_res + m * H + hh, mask=m_h, other=0.0).to(tl.float32)
            acc += c_row[:, None] * r_m[None, :]

        xv = tl.load(row_x + hh, mask=m_h, other=0.0).to(tl.float32)
        out = xv[None, :] * pm[:, None] + acc

        tl.store(row_out + mm[:, None] * H + hh[None, :],
                 out.to(tl.bfloat16),
                 mask=m_m[:, None] & m_h[None, :])


def _pick_block_h(h: int, cap: int = 4096) -> int:
    """挑 H 方向的分块大小：直接取 >= h 的最小 2 的幂，一发覆盖整个 H。

    早先这里挑的是"能整除 h 的最大 2 的幂"（h=1280 时是 256），想法是让每个块
    都满、零 padding 浪费。**实测证明这是错的方向**，昇腾 910B3 上扫出来：

        BLOCK_H   循环次数   v1(ms)    speedup
            256        5     1.2847      1.75x
            512        3     1.0697      2.08x
           1024        2     0.8724      2.56x
           2048        1     0.8201      2.71x   <- next_power_of_2(1280)

    单调递增。原因是 MTE2 每次搬运有固定开销，块小了传输次数就多，固定开销占比
    暴涨；被 mask 掉的 lane 本来就不发起访存，padding 只浪费向量槽位、并不多搬
    字节。**大块传输 + 一点 padding 浪费，好过零浪费 + 小块传输。**

    cap 是防 h 极大时累加器把片上存储撑爆（tile 是 (BLOCK_M, BLOCK_H) fp32）。
    本题 h=1280 -> 2048，够不到 cap；cap 这个值在更大的 h 上没实测过。
    """
    return min(triton.next_power_of_2(h), cap)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # 调参用的环境变量覆盖 —— 在 __init__ 里读一次，不放 forward 里，
        # 免得给被计时的路径添哪怕一点点 python 开销。
        #
        #   KS_BLOCK_H     H 方向分块大小（必须是 2 的幂；mask 兜底，任何值都正确）
        #   KS_NUM_WARPS   传给 kernel 的 num_warps
        #   KS_NUM_STAGES  传给 kernel 的 num_stages（有了 H 循环才有作用对象）
        #
        # 三个都不设时行为跟没有这段代码完全一样：BLOCK_H 由 _pick_block_h 挑，
        # num_warps/num_stages 用后端默认值。
        self._block_h_override = _env_int("KS_BLOCK_H")
        self._num_warps = _env_int("KS_NUM_WARPS")
        self._num_stages = _env_int("KS_NUM_STAGES")
        tuned = {k: v for k, v in (
            ("BLOCK_H", self._block_h_override),
            ("num_warps", self._num_warps),
            ("num_stages", self._num_stages),
        ) if v is not None}
        if tuned:
            print(f"[mhc_post] 环境变量覆盖生效: {tuned}")

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        n0, n1, mhc_mult, h = residual.shape

        # kernel 里的取址算术是按标准行优先连续布局硬编码的，非连续输入会被
        # 悄悄读错位置而不报错。get_inputs() 现造的张量天然连续，这里加上是
        # 防真遇到非连续输入时的保险，不是当前测试路径需要的。
        if not x.is_contiguous():
            x = x.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        if not post_layer_mix.is_contiguous():
            post_layer_mix = post_layer_mix.contiguous()
        if not comb_res_mix.is_contiguous():
            comb_res_mix = comb_res_mix.contiguous()

        output = torch.empty((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=x.device)

        grid = (n0 * n1,)
        block_h = self._block_h_override
        if block_h is None:
            block_h = _pick_block_h(h)

        # num_warps / num_stages 只在显式设了环境变量时才传，不设就用后端默认值
        opts = {}
        if self._num_warps is not None:
            opts["num_warps"] = self._num_warps
        if self._num_stages is not None:
            opts["num_stages"] = self._num_stages

        _mhc_post_kernel[grid](
            x, residual, post_layer_mix, comb_res_mix,
            output,
            H=h, M=mhc_mult,
            BLOCK_H=block_h,
            BLOCK_M=triton.next_power_of_2(mhc_mult),
            **opts,
        )
        return output


n0 = 2
n1 = 4096
h = 1280
mhc_mult = 4


def generate_mhc_post_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int
) -> dict[str, torch.Tensor]:
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32)

    o_grad = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16)
    return [x, residual, post_layer_mix, comb_res_mix, o_grad]


def get_inputs():
    _ks_bootstrap()
    x, residual, post_layer_mix, comb_res_mix, o_grad = generate_mhc_post_test_data(n0, n1, h, mhc_mult)
    return [x, residual, post_layer_mix, comb_res_mix]


def get_init_inputs():
    _ks_bootstrap()
    return []
