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


@triton.jit
def _fused_moe_kernel(
):
    pass


class ModelNew(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize

        # ⚠️ 参数名必须逐字保持 w1 / w2，形状也要和 v0 一致。
        #    auto_bench.py L519 的 load_state_dict 失败是**静默**的 ——
        #    改名不会报错，只会让这里的随机权重参与计算，然后数值对拍莫名挂掉。
        # w1: gate+up fused projection  [E, 2*intermediate, hidden]
        self.w1 = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        # w2: down projection  [E, hidden, intermediate]
        self.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, H]  float16
        router_logits: torch.Tensor,   # [T, E]  float32
    ) -> torch.Tensor:
        # 返回: [T, H]  float16
        #
        # 优化靶子是 v0 里 `for e in range(num_experts)` + `if not mask.any()`
        # 带来的 **8 次 host 同步**和 16 次布尔索引 gather/scatter，
        # 不是算力（全部 GEMM 才 ~8.2 MFLOP）。
        # 路由段（softmax + topk + renormalize）是 grouped_topk 的严格子集，可复用。
        # 详见 v0/fused_moe.py 顶部的 KS-PORT 说明。
        raise NotImplementedError


def get_inputs():
    _ks_bootstrap()
    # hidden_states: [num_tokens, hidden_size], float16
    # router_logits:  [num_tokens, num_experts], float32
    num_tokens, hidden_size, num_experts = 83, 128, 8
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    return [hidden_states, router_logits]


def get_init_inputs():
    _ks_bootstrap()
    # num_experts, top_k, hidden_size, intermediate_size
    return [8, 2, 128, 64]
