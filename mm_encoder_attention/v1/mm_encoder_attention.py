import torch
import torch.nn as nn
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


@triton.autotune(
    # [KS-PORT] 和 flex_attention 同一套账：寄存器按线程分配，线程是唯一的扩容手段。
    # C500 实测 regsPerBlock=131072、maxThreadsPerBlock=1024、warpSize=64。
    # flex_attention 上实测三个候选的 n_regs 分别是 171 / 158 / 156，n_spills 全 0，
    # 本 kernel 的 tile 尺寸完全相同（多的只是 batch 维在 grid 上），所以沿用同一组候选。
    # num_stages 不调 —— 本 kernel 无循环，软件流水无从谈起。
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=[],
)
@triton.jit
def _mm_encoder_attention_kernel(
    q_ptr,            # [BATCH, Q_LEN,  NUM_HEADS * HEAD_SIZE]  fp16
    k_ptr,            # [BATCH, KV_LEN, NUM_HEADS * HEAD_SIZE]  fp16
    v_ptr,            # [BATCH, KV_LEN, NUM_HEADS * HEAD_SIZE]  fp16
    out_ptr,          # [BATCH, Q_LEN,  NUM_HEADS * HEAD_SIZE]  fp16  ← 与 q 同布局
    SCALE: tl.constexpr,
    Q_LEN: tl.constexpr,          # 83
    KV_LEN: tl.constexpr,         # 83
    NUM_HEADS: tl.constexpr,      # 8
    HEAD_SIZE: tl.constexpr,      # 64（已是 2 的幂，这一维不需要 mask）
    BLOCK_M: tl.constexpr,        # next_pow2(Q_LEN)  = 128
    BLOCK_N: tl.constexpr,        # next_pow2(KV_LEN) = 128
    IS_CAUSAL: tl.constexpr,      # 本题 False（全连接注意力）
    K_CONTIG: tl.constexpr,       # K 是否必须按连续布局载入，见 _prefers_contiguous_k
):
    # grid = (BATCH, NUM_HEADS)：一个 program 负责一个 (batch, head)，一趟算完。
    # **刻意不写 for 循环遍历 K/V 分块** —— 经典 flash-attention 那种
    # "2D tile 作为 loop-carried 变量 + 循环体内规约" 正是沐曦上 make_ttgir
    # 段错误的触发条件（见 hc_split_sinkhorn/docs/）。seq=83 padding 到 128
    # 一整块就装得下，无循环即无雷。
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)          # query 位置
    offs_n = tl.arange(0, BLOCK_N)          # key/value 位置
    offs_d = tl.arange(0, HEAD_SIZE)        # head 内维度

    m_mask = offs_m < Q_LEN
    n_mask = offs_n < KV_LEN

    # 布局 [B, S, H*D] 连续，把 H*D 拆开看就是 [B, S, H, D]（同一块内存）：
    #   元素 (b, s, h, d) 的偏移 = b*(S*H*D) + s*(H*D) + h*D + d
    # 所以 batch 内的寻址和 flex_attention 完全一样，只是多一个 batch 基址。
    # 注意 K/V 用的是 KV_LEN，和 Q 的 batch stride 可能不同。
    stride_t = NUM_HEADS * HEAD_SIZE
    head_off = pid_h * HEAD_SIZE
    q_base = pid_b * Q_LEN * stride_t + head_off
    kv_base = pid_b * KV_LEN * stride_t + head_off

    # Q: [BLOCK_M, HEAD_SIZE]
    q = tl.load(q_ptr + q_base + offs_m[:, None] * stride_t + offs_d[None, :],
                mask=m_mask[:, None], other=0.0)

    # [KS-PORT] 两块卡在这里要求**相反**，用 constexpr 分支各走各的：
    #   * 沐曦：对布局转换比较脆，能不转就不转 —— 直接按转置布局读成
    #     [HEAD_SIZE, BLOCK_N]，省掉 tl.trans 的 layout conversion。
    #   * 昇腾：不接受末维非连续的 load，转置读直接编译失败
    #         'hivm.hir.load' op Unsupported op for finding the root alloc.
    #     最小复现验过这和 tl.dot 无关 —— 转置载入哪怕只 store 出去也一样挂，
    #     而"行跨步、末维连续"的 load 完全正常。只能自然布局读入再 tl.trans。
    # K_CONTIG 是 tl.constexpr，**编译期就折叠掉**，没被选中的那支根本不进 IR
    # ——和上面 IS_CAUSAL 的用法一样。昇腾上配对实测过：带分支 1.47x、
    # 写死昇腾路径 1.47x，逐轮差异 0.5% 远小于本批噪声 3.6~4.4%，开销为零。
    if K_CONTIG:
        k = tl.load(k_ptr + kv_base + offs_n[:, None] * stride_t + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0)     # [BLOCK_N, HEAD_SIZE]
        k_t = tl.trans(k)                                # [HEAD_SIZE, BLOCK_N]
    else:
        k_t = tl.load(k_ptr + kv_base + offs_d[:, None] + offs_n[None, :] * stride_t,
                      mask=n_mask[None, :], other=0.0)   # [HEAD_SIZE, BLOCK_N]

    # V: [BLOCK_N, HEAD_SIZE]
    v = tl.load(v_ptr + kv_base + offs_n[:, None] * stride_t + offs_d[None, :],
                mask=n_mask[:, None], other=0.0)

    s = tl.dot(q, k_t, out_dtype=tl.float32)          # [BLOCK_M, BLOCK_N] fp32
    s = s * SCALE

    # 只屏蔽 padding **列**：softmax 沿 axis=1 规约，列会污染每一个有效行；
    # 行不会互相污染，padding 行的垃圾结果由 store 的 m_mask 丢掉。
    # 额外屏蔽行反而会制造整行 -inf → softmax 出 NaN。
    visible = n_mask[None, :]
    if IS_CAUSAL:                                     # 本题 False，编译期整个砍掉
        visible = visible & (offs_n[None, :] <= offs_m[:, None])
    s = tl.where(visible, s, float("-inf"))

    # softmax（沿 axis=1，全程 fp32）。不能用 tl.softmax —— 它没有 axis 参数，
    # 且规约方向写死为 axis=0。
    s = s - tl.max(s, axis=1)[:, None]
    p = tl.exp(s)
    p = p / tl.sum(p, axis=1)[:, None]                # [BLOCK_M, BLOCK_N]

    # p 降回 fp16 走硬件矩阵指令，累加仍在 fp32
    acc = tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)   # [BLOCK_M, HEAD_SIZE]

    # 输出与 q 同布局，直接按 (b, m, h, d) 写回 —— v0 末尾
    # `out.transpose(1, 2).reshape(bsz, q_len, -1)` 的那次连续化拷贝就省掉了
    tl.store(out_ptr + q_base + offs_m[:, None] * stride_t + offs_d[None, :],
             acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])


def _prefers_contiguous_k():
    """K 是否必须按连续布局载入（即末维不能带大跨步）。只在 __init__ 调一次。

    [KS-PORT] 开关按**能力**命名而不是芯片名（K_CONTIG 而非 IS_ASCEND）：
    再来一块卡时是去挑已有开关的组合，而不是每个文件都新增一支。

    [KS-PORT] 探测逻辑必须待在函数体里 —— auto_bench.py L74 的
    _filter_module_ast() 只保留 Import / ClassDef / FunctionDef / 字面量赋值，
    模块级的 try/except 是 ast.Try，会被整个丢弃（和 _ks_bootstrap 同一个理由）。

    判据用 triton.backends.backends：这是唯一能区分厂商版 triton 的东西，
    __version__ 和 __file__ 都区分不出来（昇腾那份也自称 3.2.0）。
    """
    try:
        import triton
        return "ascend" in triton.backends.backends
    except Exception:
        return False


class ModelNew(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / (head_size ** 0.5)
        # 后端分支在构造时定死。auto_bench 只计时 forward，__init__ 不进计时路径，
        # 所以这次探测连"很小的开销"都算不上 —— 是零。
        self.k_contig = _prefers_contiguous_k()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # query/key/value: [bsz, seq_len, num_heads * head_size]  float16
        # 返回: [bsz, q_len, num_heads * head_size]  float16
        #
        # num_kv_heads == num_heads == 8，不需要实现 GQA。
        bsz, q_len = query.shape[:2]
        kv_len = key.shape[1]
        assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()

        out = torch.empty(bsz, q_len, self.num_heads * self.head_size,
                          device=query.device, dtype=query.dtype)

        _mm_encoder_attention_kernel[(bsz, self.num_heads)](
            query, key, value, out,
            SCALE=self.scale,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            NUM_HEADS=self.num_heads,
            HEAD_SIZE=self.head_size,
            BLOCK_M=triton.next_power_of_2(q_len),      # 83 -> 128
            BLOCK_N=triton.next_power_of_2(kv_len),     # 83 -> 128
            IS_CAUSAL=False,                            # 本题是全连接注意力
            K_CONTIG=self.k_contig,
        )
        return out


def get_inputs():
    _ks_bootstrap()
    # query: [bsz, q_len, num_heads * head_size], float16
    # key:   [bsz, kv_len, num_kv_heads * head_size], float16
    # value: [bsz, kv_len, num_kv_heads * head_size], float16
    bsz, seq_len, num_heads, head_size, dtype = 2, 83, 8, 64, torch.float16
    hidden = num_heads * head_size
    query = torch.randn(bsz, seq_len, hidden, dtype=dtype)
    key = torch.randn(bsz, seq_len, hidden, dtype=dtype)
    value = torch.randn(bsz, seq_len, hidden, dtype=dtype)
    return [query, key, value]


def get_init_inputs():
    _ks_bootstrap()
    # num_heads, head_size, num_kv_heads
    return [8, 64, 8]
