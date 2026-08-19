import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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


# ---------------------------------------------------------------------------
# [KS-PLAN] 分两步走，本文件先做第一步。
#
#   **第一步（本文件）**：MLM head 保持 torch（厂商 GEMM 本来就最优），只用一个
#   Triton kernel 融合 `log1p(relu(x))` + 分段 pooling。收益来自：
#     * 干掉 `seq_lens.tolist()` 这**一次 host 同步**（v0 L110）；
#     * 干掉 pooling 那个 4 次迭代的 Python 循环；
#     * `[83, 30522]` fp32（9.7 MB）只读一遍就出结果，而不是 log1p/relu 各写读一遍。
#   预期 1.3~1.6x。工作量小一个数量级，先保证"完成"。
#
#   **第二步（暂不做）**：把 decoder GEMM 也融进来，中间结果彻底不落地。
#     grid = (cdiv(V, BLOCK_V),) —— **一个 program 覆盖全部 4 条序列**，
#     而不是 (S, V块)。因为 4 条序列共用同一份 decoder.weight：按 (S, V块) 切
#     会让同一块权重被 4 个 program 各读一遍，89.5 MB 变 358 MB。
#     每个 program：acc[128, BLOCK_V] 沿 K=768 累加 -> 加 bias -> 对 4 条序列各做
#     一次 masked max -> log1p(relu(·)) -> 写出。83 个 token padding 到 128 一次装下，
#     M 维不需要循环（L 最大 25）。
#     LayerNorm 沿 768 整行规约，跨了 dense 输出的全部 N 维，是天然的 program 间
#     依赖 —— 所以 dense+GELU+LN 必须是另一个 kernel，且那个 kernel **不能切 N**
#     （grid=(cdiv(83,BLOCK_M),)，只切 K），LN 才能在 program 内完成。
#     那一段只占 2.5% 的算力，别过度优化。
#
#   **一个 v0 注释里没算到的杠杆**：地板 ~245µs 是按 fp32 权重 93.8 MB / 382 GB/s
#   算的。若沿用 fused_moe 的 [KS-CACHE] 把 decoder.weight 惰性预降成 fp16 并缓存，
#   访存减半 -> 地板 ~123µs，还能吃上 tensor core。粗估精度：LN 之后 h ~ N(0,1)，
#   decoder 输出量级约 0.6，fp16 相对精度 1e-3 -> 绝对误差 ~6e-4，log1p 在该处
#   斜率 0.63，max 不放大 —— 离 atol=1e-2 有一个数量级余量。**但这是估算，必须实测**。
# ---------------------------------------------------------------------------


# 先写死跑通，之后再考虑挂 autotune。fused_moe 的教训：Autotuner 包装层每次调用
# 约 15.3 µs，本题只有一个 kernel、且是访存受限，旋钮的收益可能还不如那笔开销。
#
# 两个 BLOCK 的取值靠 **乘积** 约束，不是各自独立：规约要 load 一个
# [BLOCK_L, BLOCK_V] 的 fp32 tile，摊到 num_warps×64 个线程上就是每线程
# BLOCK_L·BLOCK_V/线程数 个寄存器（架构上限约 255）：
#
#     BLOCK_L  BLOCK_V   tile   寄存器/线程(4 warps)   L=25 要几轮
#           8      512    16K            16                4
#          32      256    32K            32                1   <- 当前取值
#          32      512    64K            64                1
#          32     1024   128K           128                1   <- 太满，不用
#
# 取 32×256：L 最大 25，BLOCK_L=32 一个 tile 就覆盖任一段，循环只跑一轮；
# BLOCK_V=256 是 1 KB 连续，合并访存绰绰有余，且 grid 有 4×120=480 个 program。
# ⚠ 别照着这张表去精调 —— fused_moe 里同类估算预测 76 寄存器、实测 spill 441，
#   这种"tile/线程数"的模型在 MACA 后端上不可信。它只用来排除明显过大的取值，
#   真要调就上 bench/check_spill.py 看实测的 n_spills。
_BLOCK_L = 32
_BLOCK_V = 256
_NUM_WARPS = 4

# ---------------------------------------------------------------------------
# [KS-FP16] decoder 这一路降半精度 —— 本题唯一真正有肉的优化。
#
#   实测依据（scratch/breakdown.py，口径同 auto_bench）：
#       dense        6.7 µs   roofline   6.8    效率 101%
#       GELU         7.6 µs   roofline   0.7      9%
#       LayerNorm   18.9 µs   roofline   0.7      4%
#       decoder    327.7 µs   roofline 272.3     83%   <- 占 forward 的 89%
#       pool         6.6 µs   roofline  27.8    421%   <- 命中 cache，比 HBM 还快
#
#   decoder 已经贴着带宽跑（83%），重排结构榨不出东西，**只能减字节数**：
#   decoder.weight 30522×768 从 fp32 的 89.4 MB 降到 fp16 的 44.7 MB，
#   中间张量 [83,30522] 同步减半，预期 327.7 -> ~164 µs。
#
#   其余候选加起来不到它的 1/5，都不做：
#     * 融合 dense+GELU+LN 省中间张量        ~1 µs
#     * 少 3 次 kernel 启动（边际 3.7 µs/次） ~11 µs
#     * 融合 decoder+pool                    ~6.6 µs（不是 roofline 估的 53 µs ——
#       那次往返命中 cache，本来就没花 HBM 的钱，见上表 pool 那行的 421%）
#
#   **只降 decoder，前面保持 fp32**：dense 才 6.7 µs，而 LayerNorm 在 fp16 下
#   算方差是有名的容易掉精度，为 19 µs 冒那个险不值得。
#
#   ⚠ 这里必须是 **bool 字面量**，不能写 `_GEMM_DTYPE = torch.float16`。
#     auto_bench.py L74 的 _filter_module_ast() 只保留 Import / ClassDef /
#     FunctionDef / **字面量**赋值四类节点，而 `torch.float16` 是 ast.Attribute
#     不是 ast.Constant，整行会被**静默丢弃** —— 表现为运行时
#     "v1 forward failed: name '_GEMM_DTYPE' is not defined"。
#     （踩过一次。同一个过滤器还会吃掉模块级的 try/except，见 _ks_bootstrap。）
#     dtype 在方法里解析，函数体内部不受那个过滤器影响。
#
#   改回 fp32 做 A/B：把下面这行改成 False。
_USE_FP16_DECODER = True
# ---------------------------------------------------------------------------


@triton.jit
def _splade_pool_kernel(
    x_ptr,              # [T, V]  fp16 或 fp32（见 [KS-FP16]）—— load 后立刻升 fp32
    seq_lens_ptr,       # [S]     int32  各段长度，和 = T
    out_ptr,            # [S, V]  fp32
    V,                  # 30522，运行时值（不是 2 的幂，必须带 mask）
    S: tl.constexpr,        # 4，段数
    BLOCK_S: tl.constexpr,  # next_pow2(S) = 4
    BLOCK_L: tl.constexpr,  # 一次处理多少行 token
    BLOCK_V: tl.constexpr,  # 一次处理多少列词表
    POOLING_MAX: tl.constexpr,  # True=max, False=sum
):
    # 关于 log1p：这里用 tl.log(1.0 + x) 而不是 tl.math.log1p。两者的差别只在
    # x 小到 1+x 舍入成 1.0 时（fp32 下 x < 6e-8），绝对误差 ~6e-8，对 atol=1e-2
    # 完全无关紧要；而 log1p 走 libdevice 映射，正是沐曦/昇腾后端容易出问题的地方
    # （参见 fused_moe 里对 tl.sigmoid 的同类顾虑）。不值得为这点精度冒可移植性的险。
    pid_s = tl.program_id(0)        # 第几条序列
    pid_v = tl.program_id(1)        # 第几个词表分块

    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    # ---- 从 seq_lens 现算本段的 [offset, offset+L)，避免 host 侧的 .tolist() ----
    # S 只有 4，整个 seq_lens 一次load 进来做前缀和，比在 host 上算完再传省一次同步。
    offs_s = tl.arange(0, BLOCK_S)
    lens = tl.load(seq_lens_ptr + offs_s, mask=offs_s < S, other=0).to(tl.int32)
    offset = tl.sum(tl.where(offs_s < pid_s, lens, 0))      # 前缀和
    L = tl.sum(tl.where(offs_s == pid_s, lens, 0))          # 本段长度

    # 累加器初值取各自运算的单位元。max 用 -inf 而不是 0：0 在本 kernel 里也
    # 恰好等价（relu 就是 max(x,0)，多一个 0 参与 max 不改结果，已逐位验证），
    # 但那个正确性是从后面的 relu 借来的，换掉 epilogue 就悄悄错了。
    if POOLING_MAX:
        acc = tl.full((BLOCK_V,), float("-inf"), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

    # L 是运行时值，所以这是运行时上界的循环。BLOCK_L=32 >= max(L)=25，
    # 实际只跑一轮 —— 但写成循环才不依赖"段长一定 <= BLOCK_L"这个假设。
    for l0 in range(0, L, BLOCK_L):
        rows = l0 + tl.arange(0, BLOCK_L)
        m = rows < L                                   # 段内有效行
        ptrs = x_ptr + (offset + rows)[:, None] * V + offs_v[None, :]
        mask = m[:, None] & mask_v[None, :]

        if POOLING_MAX:
            # 单调性等价：max_L(log1p(relu(z))) == log1p(relu(max_L(z)))，
            # 所以这里只在**原始 logits** 上取 max，激活留到循环外做一次。
            tile = tl.load(ptrs, mask=mask, other=float("-inf")).to(tl.float32)
            acc = tl.maximum(acc, tl.max(tile, axis=0))
        else:
            # sum 与非线性不可交换，必须逐元素激活后再累加。
            tile = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(tl.log(1.0 + tl.maximum(tile, 0.0)), axis=0)

    # max 分支的激活挪到这里：只作用在 [BLOCK_V] 上，而不是 [L, BLOCK_V]，
    # 省 18~25 倍的逐元素运算。sum 分支在循环内已经算过了。
    if POOLING_MAX:
        acc = tl.log(1.0 + tl.maximum(acc, 0.0))

    tl.store(out_ptr + pid_s * V + offs_v, acc, mask=mask_v)


class ModelNew(nn.Module):
    """SPLADESparsePooler: MLM head logits → ReLU log(1+x) pooled over sequence (max or sum)."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pooling: str = "max",
    ):
        super().__init__()
        # ⚠️ 子模块名必须逐字保持 dense / layer_norm / decoder，形状也要一致。
        #    auto_bench.py L519 的 load_state_dict 失败是**静默**的 ——
        #    改名不会报错，只会让随机初始化的权重参与计算，然后数值对拍莫名挂掉。
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling

    def _decoder_weights(self, dtype):
        """[KS-CACHE] 惰性产出降精度后的 decoder 权重，按 Tensor._version 失效。

          * 失效判据用 _version：load_state_dict 走 param.copy_()，是原地写，会顶
            _version，所以 auto_bench 灌完权重后的第一次 forward 一定重建缓存，
            不会拿着 __init__ 里的随机权重去算。
          * **不能** register_buffer：那会往 state_dict 里塞多余的键，
            auto_bench.py L519 的 load_state_dict 会因 missing key 失败，
            而失败是静默的。裸 Tensor 属性走 object.__setattr__，不进 _buffers。
          * 必须惰性（不能放 __init__）：load_state_dict 发生在 __init__ 之后。
        """
        w, b = self.decoder.weight, self.decoder.bias
        key = (w._version, b._version, dtype, w.device)
        if getattr(self, "_wkey", None) != key:
            with torch.no_grad():   # 参数 requires_grad=True，别挂进计算图
                self._dw = w.to(dtype)
                self._db = b.to(dtype)
            self._wkey = key
        return self._dw, self._db

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> list:
        # hidden_states: [83, 768] float32   seq_lens: [4] int32，和为 83
        #
        # dense / GELU / LayerNorm 保持 fp32 —— 它们合计只有 33 µs，而 LayerNorm
        # 在 fp16 下算方差容易掉精度，不值得冒险（见 [KS-FP16]）。
        h = self.layer_norm(self.act(self.dense(hidden_states)))     # [T, H] fp32

        # decoder 走半精度：这一步占 forward 的 89%，且已贴着带宽跑，
        # 唯一的出路是把 89.4 MB 的权重读减半。GEMM 内部仍是 fp32 累加。
        # dtype 在这里解析（函数体内），不能放模块级 —— 见 [KS-FP16] 的说明。
        gemm_dtype = torch.float16 if _USE_FP16_DECODER else torch.float32
        dw, db = self._decoder_weights(gemm_dtype)
        x = F.linear(h.to(gemm_dtype), dw, db)                       # [T, V] fp16
        T, V = x.shape

        # S 是**形状**，取它不需要同步；取 seq_lens 的**值**才需要（那正是 v0
        # 里 .tolist() 的问题）。所以 grid 能在 host 上算，段边界在 kernel 里算。
        S = seq_lens.shape[0]

        x = x.contiguous()
        seq_lens = seq_lens.contiguous()
        # ⚠ 输出 dtype 跟 **hidden_states**，不能跟 x —— x 现在是 fp16，而 v0
        #   返回的是 fp32；torch.allclose 要求 dtype 一致，跟错了直接判类型不匹配。
        out = torch.empty((S, V), dtype=hidden_states.dtype, device=x.device)

        grid = lambda META: (S, triton.cdiv(V, META["BLOCK_V"]))   # noqa: E731
        _splade_pool_kernel[grid](
            x, seq_lens, out,
            V,
            S=S,
            BLOCK_S=triton.next_power_of_2(S),
            BLOCK_L=_BLOCK_L,
            BLOCK_V=_BLOCK_V,
            POOLING_MAX=(self.pooling == "max"),
            num_warps=_NUM_WARPS,
        )

        # ⚠️ 必须返回 **list**（S 个 [V] 张量），不能是 tuple ——
        #    auto_bench.py L328 的 compare_values 会先比类型再逐项比。
        #    unbind 出来的是 out 的视图，不复制。
        return list(out.unbind(0))


def get_inputs():
    _ks_bootstrap()
    # hidden_states: [83, 768], float32
    # seq_lens: [4], int32，和为 83
    seq_lens = torch.tensor([20, 25, 18, 20], dtype=torch.int32)
    hidden_states = torch.randn(83, 768)
    return [hidden_states, seq_lens]


def get_init_inputs():
    _ks_bootstrap()
    # hidden_size=768, vocab_size=30522, pooling="max"
    return [768, 30522, "max"]
