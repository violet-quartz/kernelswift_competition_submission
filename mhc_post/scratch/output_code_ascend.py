# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import torch_npu
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream
import torch_npu
has_initialized = False
import torch_npu._inductor.npu_triton_heuristics as triton_heuristics
from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/rn/crnn76frr5tn25dtl7vpcbih3lbw55er4xa6fkclx65swdnp63l2.py
# Topologically Sorted Source Nodes: [float_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
from torch_npu._inductor import npu_triton_heuristics as triton_heuristics
from torch_npu._inductor.npu_triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints=[41943040], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'x0_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B3', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'backend_hash': '9026E7E0DC95307C1352589CA1295B692A0EEE0AEFB73FD9D9450CFA787B270C', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simd', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, x0_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
        tmp1 = tmp0.to(tl.float32)
        tl.store(out_ptr0 + (x0), tmp1, x0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/sx/csxlx2najlbu2pvjo6p5q7z67ugc7fox53kn2lyhg55s42fnn3j7.py
# Topologically Sorted Source Nodes: [mul, add, bfloat16], Original ATen: [aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   add => add
#   bfloat16 => convert_element_type_2
#   mul => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_2, %arg3_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %view_3), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.bfloat16), kwargs = {})
# SchedulerNodes: [SchedulerNode(name='op2')]

triton_poi_fused__to_copy_add_mul_1 = async_compile.triton('triton_poi_fused__to_copy_add_mul_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
from torch_npu._inductor import npu_triton_heuristics as triton_heuristics
from torch_npu._inductor.npu_triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints=[8192, 4, 1280], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'z0_numel': 'i32', 'y1_numel': 'i32', 'x2_numel': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B3', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_1', 'mutated_arg_names': [], 'backend_hash': '9026E7E0DC95307C1352589CA1295B692A0EEE0AEFB73FD9D9450CFA787B270C', 'split_axis': [0], 'tiling_axis': [0, 1, 2], 'no_loop_axis': [2], 'axis_names': ['z0', 'y1', 'x2'], 'low_dims': {2}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.bfloat16, 'dual_reduction': False, 'npu_kernel_type': 'simd', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, z0_numel, y1_numel, x2_numel, Z0BLOCK : tl.constexpr, Z0BLOCK_SUB : tl.constexpr, Y1BLOCK_SUB : tl.constexpr):
    x2_numel = 1280
    X2BLOCK_SUB: tl.constexpr = 1280
    z0_offset = tl.program_id(0) * Z0BLOCK
    base_z0= tl.arange(0, Z0BLOCK_SUB)
    loops_z0 = (Z0BLOCK + Z0BLOCK_SUB - 1) // Z0BLOCK_SUB
    base_y1= tl.arange(0, Y1BLOCK_SUB)
    loops_y1 = (y1_numel + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    for loop_z0 in range(loops_z0):
        z0 = z0_offset + (loop_z0 * Z0BLOCK_SUB) + base_z0[:,None,None]
        z0_mask = z0 < min(Z0BLOCK+z0_offset, z0_numel)
        for loop_y1 in range(loops_y1):
            y1 = (loop_y1 * Y1BLOCK_SUB) + base_y1[None,:,None]
            y1_mask = y1 < y1_numel
            x2 = base_x2[None,None,:]
            tmp0 = tl.load(in_ptr0 + (x2 + 1280*z0), z0_mask)
            tmp2 = tl.load(in_ptr1 + (y1 + 4*z0), y1_mask & z0_mask)
            tmp4 = tl.load(in_ptr2 + (x2 + 1280*y1 + 5120*z0), y1_mask & z0_mask)
            tmp1 = tmp0.to(tl.float32)
            tmp3 = tmp1 * tmp2
            tmp5 = tmp3 + tmp4
            tmp6 = tmp5.to(tl.float32)
            tl.store(out_ptr0 + (x2 + 1280*y1 + 5120*z0), tmp6, y1_mask & z0_mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    with torch.npu.utils.device(0):
        torch.npu.set_device(0)
        buf0 = empty_strided((2, 4096, 4, 1280), (20971520, 5120, 1280, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [float_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(arg0_1, buf0, 41943040, stream=stream0)
        del arg0_1
        buf1 = empty_strided((8192, 4, 1280), (5120, 1280, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [term2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg1_1, (8192, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf0, (8192, 4, 1280), (5120, 1280, 1), 0), out=buf1)
        del arg1_1
        del buf0
        buf2 = empty_strided((2, 4096, 4, 1280), (20971520, 5120, 1280, 1), device='npu', dtype=torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul, add, bfloat16], Original ATen: [aten.mul, aten.add, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mul_1.run(arg2_1, arg3_1, buf1, buf2, 8192, 4, 1280, stream=stream0)
        del arg2_1
        del arg3_1
        del buf1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 4096, 4, 1280), (20971520, 5120, 1280, 1), device='npu:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((2, 4096, 4, 4), (65536, 16, 4, 1), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((2, 4096, 1280), (5242880, 1280, 1), device='npu:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((2, 4096, 4, 1), (16384, 4, 1, 1), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
