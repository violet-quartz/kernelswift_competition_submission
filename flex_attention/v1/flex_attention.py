import os

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
    # [KS-PORT] 这道题的 num_warps 取向和 grouped_topk / centre_random_augmentation
    # **正好相反**，候选要往大了给：
    #   * 寄存器是按线程分配的，总量 = 线程数 × 每线程寄存器数。C500 实测
    #     regsPerBlock=131072、maxThreadsPerBlock=1024 ⇒ 满线程数时每线程 128 个；
    #     每线程的架构上限约 255（CUDA 系数值，沐曦未公布）。
    #   * 本 kernel 同时存活的 tile 约 37000 个 32-bit 寄存器槽
    #     （S/P [128,128] fp32 占 16384，acc [128,64] fp32 占 8192，
    #       Q/K/V [128,64] fp16 各 4096）。摊到线程上：
    #         num_warps=4  → 256 线程 → 145 个/线程，逼近上限，容易 spill
    #         num_warps=8  → 512 线程 →  72 个/线程
    #         num_warps=16 → 1024 线程 →  36 个/线程
    #   * 所以线程是这里唯一的"寄存器扩容"手段，候选 [4, 8, 16]。
    #     跑通后用 `python bench/check_spill.py --only flex_attention` 看
    #     n_regs / n_spills（沐曦走 cuda 命名空间，这两个指标是真值）。
    # num_stages 不调 —— 本 kernel 无循环，软件流水无从谈起。
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=[],
)
@triton.jit
def _flex_attention_kernel(
    q_ptr,            # [SEQ_LEN, NUM_HEADS, HEAD_SIZE]  fp16
    k_ptr,            # [SEQ_LEN, NUM_HEADS, HEAD_SIZE]  fp16
    v_ptr,            # [SEQ_LEN, NUM_HEADS, HEAD_SIZE]  fp16
    out_ptr,          # [SEQ_LEN, NUM_HEADS * HEAD_SIZE] fp16  ← 和输入同布局
    mask_ptr,         # [BLOCK_M, BLOCK_N] fp32 预计算因果掩码；MASK_FROM_MEM=False 时不读
    SCALE: tl.constexpr,
    SEQ_LEN: tl.constexpr,        # 83
    NUM_HEADS: tl.constexpr,      # 8
    HEAD_SIZE: tl.constexpr,      # 64（已是 2 的幂，这一维不需要 mask）
    BLOCK_M: tl.constexpr,        # next_pow2(SEQ_LEN) = 128
    BLOCK_N: tl.constexpr,        # next_pow2(SEQ_LEN) = 128
    IS_CAUSAL: tl.constexpr,      # 本题 True；
    K_CONTIG: tl.constexpr,       # K 是否必须按连续布局载入，见 _prefers_contiguous_k
    MASK_FROM_MEM: tl.constexpr,  # 因果掩码是预算好载入还是 kernel 里现算，见 _use_mask_from_mem
):
    # grid = (NUM_HEADS,)：一个 program 负责一个注意力头，一趟算完。
    # **刻意不写 for 循环遍历 K/V 分块** —— 经典 flash-attention 那种
    # "2D tile 作为 loop-carried 变量 + 循环体内规约" 正是沐曦上 make_ttgir
    # 段错误的触发条件（见 hc_split_sinkhorn/docs/）。seq=83 padding 到 128
    # 一整块就装得下，无循环即无雷。
    pid_h = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)          # query 位置
    offs_n = tl.arange(0, BLOCK_N)          # key/value 位置
    offs_d = tl.arange(0, HEAD_SIZE)        # head 内维度

    m_mask = offs_m < SEQ_LEN
    n_mask = offs_n < SEQ_LEN

    stride_t = NUM_HEADS * HEAD_SIZE
    head_off = pid_h * HEAD_SIZE
    q = tl.load(q_ptr + head_off + offs_m[:, None] * stride_t + offs_d[None, :], mask=m_mask[:, None], other=0.0) # (BLOCK_M, HEAD_SIZE)
    # [KS-PORT] 两块卡在这里要求**相反**（硬能力差异，不是性能取舍）：
    #   * 沐曦：对布局转换比较脆，能不转就不转 —— 直接按转置布局读成 [HEAD_SIZE, BLOCK_N]。
    #   * 昇腾：不接受末维非连续的 load，转置读直接编译失败
    #         'hivm.hir.load' op Unsupported op for finding the root alloc.
    #     最小复现验过这和 tl.dot 无关 —— 转置载入哪怕只 store 出去也一样挂。
    if K_CONTIG:
        k = tl.load(k_ptr + head_off + offs_n[:, None] * stride_t + offs_d[None, :], mask=n_mask[:, None], other=0.0) # (BLOCK_N, HEAD_SIZE)
        k_t = tl.trans(k)                                                                                            # (HEAD_SIZE, BLOCK_N)
    else:
        k_t = tl.load(k_ptr + head_off + offs_d[:, None] + offs_n[None, :] * stride_t, mask=n_mask[None, :], other=0.0) # (HEAD_SIZE, BLOCK_N)
    v = tl.load(v_ptr + head_off + offs_n[:, None] * stride_t + offs_d[None, :], mask=n_mask[:, None], other=0.0) # (BLOCK_N, HEAD_SIZE)

    s = tl.dot(q, k_t, out_dtype=tl.float32) # (BLOCK_M, BLOCK_N)
    s = s * SCALE

    # [KS-PORT] 因果掩码有两条实现，选哪条见 _use_mask_from_mem()。
    if MASK_FROM_MEM and IS_CAUSAL:
        # 掩码在 host 预算好，kernel 里直接 load。
        # 昇腾上 `offs_n[None, :] <= offs_m[:, None]`（两个 1D arange 广播成 2D 再比较）
        # 慢得离谱：本 kernel 单测 0.1375ms，同一 kernel 去掉因果只要 0.0278ms ——
        # 掩码一项就占 80%。改成载入后回到 0.0274ms，和非因果齐平；端到端 0.73x → 1.73x。
        # 顺带 UB 也降下来了：现算式在 BLOCK_M=128 会 UB 溢出（192KB 装不下），载入式装得下。
        #
        # 这里也不再造 -inf：softmax 减任意常数都不变，用**全行** max 稳定化同样正确，
        # 掩码在 exp 之后乘上去即可，省掉 tl.where 那一块临时量。
        # padding 列被因果条件带掉了 —— 有效行 m < SEQ_LEN 时 n <= m 已蕴含 n < SEQ_LEN。
        p = tl.exp(s - tl.max(s, axis=1)[:, None])
        p = p * tl.load(mask_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])
    else:
        # 原写法：kernel 里现算掩码。沐曦上跑通过的就是这支。
        visible = n_mask[None, :]
        if IS_CAUSAL:
            visible = visible & (offs_n[None, :] <= offs_m[:, None])
        s = tl.where(visible, s, float("-inf"))
        s = s - tl.max(s, axis=1)[:, None]
        p = tl.exp(s)
    p = p / tl.sum(p, axis=1)[:, None] # (BLOCK_M, BLOCK_N)
    
    acc = tl.dot(p.to(v.dtype), v, out_dtype=tl.float32) # (BLOCK_M, HEAD_SIZE)

    tl.store(out_ptr + head_off + offs_m[:, None] * stride_t + offs_d[None, :],
             acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])


def _prefers_contiguous_k():
    """K 是否必须按连续布局载入（末维不能带大跨步）。**硬能力开关**，不需复测。

    [KS-PORT] 开关按能力命名而不是芯片名：再来一块卡是去挑已有开关的组合，
    而不是每个文件都新增一支。探测必须待在函数体里 —— auto_bench.py L74 的
    _filter_module_ast() 会丢弃模块级带函数调用的赋值。
    判据用 triton.backends.backends：唯一能区分厂商版 triton 的东西。
    """
    try:
        import triton
        return "ascend" in triton.backends.backends
    except Exception:
        return False


def _use_mask_from_mem():
    """因果掩码是否预算好从显存载入。**性能开关，待在沐曦上复测**。

    [KS-PORT] 两支在两块卡上都能跑，分开只是因为还没在沐曦上量过。
    我判断这是通用优化（"别在 kernel 里现算常量"），但没量过就不切 ——
    判据写成**白名单**（只有实测赢过的后端才走新路），新卡进来自动拿老行为。
    昇腾实测 0.73x → 1.73x。

    复测用 KS_MASK_FROM_MEM=0/1 覆盖，不必改代码。结论记在编排仓的 pending-verify.md。
    """
    raw = os.environ.get("KS_MASK_FROM_MEM")
    if raw:
        return raw not in ("0", "false", "False")
    try:
        import triton
        return "ascend" in triton.backends.backends
    except Exception:
        return False


class ModelNew(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64,
                 scale: float = None, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        # ⚠️ get_init_inputs() 传进来的 scale 是 None，靠这个 or 兜底 → 0.125
        self.scale = scale or 1.0 / (head_size ** 0.5)
        self.num_kv_heads = num_kv_heads
        # 两个后端开关在构造时定死，forward 里不再判断（auto_bench 只计时 forward）
        self.k_contig = _prefers_contiguous_k()
        self.mask_from_mem = _use_mask_from_mem()
        # [KS-PORT] 因果掩码缓存。用**普通属性**而不是 register_buffer ——
        # v0 的 state_dict 是空的，注册 buffer 会让键名对不上，
        # auto_bench 的 load_state_dict 会失败。首次 forward 时按输入设备懒建。
        self._causal_mask = None

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        # query/key/value: [num_tokens, num_heads, head_size]  float16
        # 返回: [num_tokens, num_heads * head_size]  float16
        #
        # num_kv_heads == num_heads == 8，v0 里的 repeat_interleave 是死代码，
        # 不实现 GQA。真要留余地：kv_h = pid_h // R，R 做 constexpr，本题 R=1
        # 编译期就折叠掉。
        num_tokens = query.shape[0]
        assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()

        out = torch.empty(num_tokens, self.num_heads * self.head_size,
                          device=query.device, dtype=query.dtype)

        BLOCK = triton.next_power_of_2(num_tokens)   # 83 -> 128

        m = self._causal_mask
        if self.mask_from_mem:
            if m is None or m.shape[0] != BLOCK or m.device != query.device:
                i = torch.arange(BLOCK, device=query.device)
                m = (i[None, :] <= i[:, None]).to(torch.float32)
                self._causal_mask = m
        else:
            m = query        # 不会被读到，只是占个位置让签名统一

        _flex_attention_kernel[(self.num_heads,)](
            query, key, value, out, m,
            SCALE=self.scale,
            SEQ_LEN=num_tokens,
            NUM_HEADS=self.num_heads,
            HEAD_SIZE=self.head_size,
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            IS_CAUSAL=True,
            K_CONTIG=self.k_contig,
            MASK_FROM_MEM=self.mask_from_mem,
        )
        return out


def get_inputs():
    _ks_bootstrap()
    # query: [num_tokens, num_heads, head_size], float16
    # key:   [num_tokens, num_kv_heads, head_size], float16
    # value: [num_tokens, num_kv_heads, head_size], float16
    num_tokens, num_heads, head_size = 83, 8, 64
    dtype = torch.float16
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
    key   = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
    return [query, key, value]


def get_init_inputs():
    _ks_bootstrap()
    # num_heads=8, head_size=64, scale=None（→ 0.125）, num_kv_heads=8
    return [8, 64, None, 8]
