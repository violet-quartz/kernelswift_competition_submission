"""v0 参考实现（torch baseline）— 源自 tasks/grouped_topk.py

【与原题的两处差异，都不改变计算本身】
  1. 加了 `_ks_bootstrap()`，并在 get_inputs / get_init_inputs 开头调用
  2. 删掉了原题末尾的 `if __name__ == "__main__"` 演示块 —— auto_bench.py L74
     的 _filter_module_ast() 本来就会把 ast.If 整个丢弃

`get_inputs()` 本身**没有改**：原题就没有硬编码 `device="cuda"`，`torch.randn(...)`
不带 device 参数默认建在 CPU 上，天然满足本仓库"v0/v1 一律返回 CPU 张量"的约定
（见 KS-PORT 说明），不需要额外处理。

`Model` 类逐字未改，也没有任何 `nn.Parameter` / `register_buffer`
—— `state_dict()` 是空的，`load_state_dict` 那步天然不会出问题。
原题在 tasks/grouped_topk.py 里留了一份。

写 v1 时要注意的两条（来自赛题背景，不是 auto_bench 强加的）：
  * **`topk_ids` 是 int32，逐位比对无容差**——不像 `topk_weights` 走 atol/rtol，
    专家 id 选错 1 个就是错。

    但**不要试图复现 `torch.topk` 的并列顺序**：它是未定义行为，不是"稳定、
    偏向靠前的下标"。2026-08-15 实测，把整行分数置成全等时 `torch.topk` 返回
    `[127, 124, 2, 3, 123, 122, 6, 121]` 这种无规律顺序，对不齐也不值得对齐。
    实际影响可忽略——`torch.randn` 的 float32 输入下，16600 行里只有 9 行
    （0.05%）存在任何重复值，而且只有并列恰好跨在选择边界上才会改变结果。

    v1 里真正需要保证的不是"和 torch 一致的 tie-break"，而是**选择本身的确定性**：
    排名法里那个下标 tie-break 子句（`g[j]==g[i] & j<i`）让"打赢"成为严格全序，
    从而保证恰好选中 k 个组；换成阈值法（`g >= 第k大`）在并列时会多选，那才是
    真正的 bug。
  * `hidden_states` **只用于 `assert hidden_states.size(0) == gating_output.size(0)`
    这一行**，数值从未参与计算——kernel 不需要读它的内容，只需要那个 batch 维度
    （如果 v1 里想保留这个校验的话）。
"""
import torch
import torch.nn as nn


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


class Model(nn.Module):
    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.size(0) == gating_output.size(0)

        if self.scoring_func == "softmax":
            scores = torch.softmax(gating_output, dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = gating_output.sigmoid()
        else:
            raise ValueError(f"Unsupported scoring_func: {self.scoring_func}")

        num_token = scores.size(0)
        experts_per_group = scores.size(-1) // self.num_expert_group

        # Select top-k groups by their best expert score
        group_scores = (
            scores.view(num_token, self.num_expert_group, -1).max(dim=-1).values
        )  # [T, n_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]  # [T, topk_group]

        # Build a mask that zeroes out experts from unchosen groups
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)  # [T, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, self.num_expert_group, experts_per_group)
            .reshape(num_token, -1)
        )  # [T, E]

        tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        topk_weights, topk_ids = torch.topk(tmp_scores, k=self.topk, dim=-1)

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def get_inputs():
    _ks_bootstrap()
    # hidden_states: [num_tokens, hidden_size], float16 — only used for batch-size check
    # gating_output: [num_tokens, num_experts], float32
    num_tokens, hidden_size, num_experts = 83, 7168, 256
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    return [hidden_states, gating_output]


def get_init_inputs():
    _ks_bootstrap()
    # topk=8, renormalize=True, num_expert_group=8, topk_group=4
    return [8, True, 8, 4]
