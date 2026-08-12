
import os
os.environ['TORCH_LOGS'] = 'output_code'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['TORCH_WARM_POOL'] = '0'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

import torch_npu._inductor.fx_passes.ascend_custom_passes
torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.post_grad_custom_post_pass = torch_npu._inductor.fx_passes.ascend_custom_passes.run_register_post_custom_passes
torch._inductor.config.fallback_random = True
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.7.1+cpu
# torch cuda version: None
# torch git version: e2d141dbde55c2a4370fac5165b0561b6af4798b


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, 4);  arg1_1 = None
        permute = torch.ops.aten.permute.default(unsqueeze, [0, 1, 3, 4, 2]);  unsqueeze = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(convert_element_type, 4);  convert_element_type = None
        permute_2 = torch.ops.aten.permute.default(permute, [0, 1, 2, 4, 3]);  permute = None
        view = torch.ops.aten.view.default(permute_2, [8192, 4, 4]);  permute_2 = None
        view_1 = torch.ops.aten.view.default(unsqueeze_1, [8192, 4, 1280]);  unsqueeze_1 = None
        bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2 = torch.ops.aten.view.default(bmm, [2, 4096, 4, 1, 1280]);  bmm = None
        permute_4 = torch.ops.aten.permute.default(view_2, [0, 1, 2, 4, 3]);  view_2 = None
        view_3 = torch.ops.aten.view.default(permute_4, [2, 4096, 4, 1280]);  permute_4 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(convert_element_type_1, -2);  convert_element_type_1 = None
        mul = torch.ops.aten.mul.Tensor(unsqueeze_2, arg3_1);  unsqueeze_2 = arg3_1 = None
        add = torch.ops.aten.add.Tensor(mul, view_3);  mul = view_3 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        return (convert_element_type_2,)
        
def load_args(reader):
    buf0 = reader.storage(None, 83886080, device=device(type='npu', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (2, 4096, 4, 1280), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 524288, device=device(type='npu', index=0))
    reader.tensor(buf1, (2, 4096, 4, 4), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 20971520, device=device(type='npu', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (2, 4096, 1280), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 131072, device=device(type='npu', index=0))
    reader.tensor(buf3, (2, 4096, 4, 1), is_leaf=True)  # arg3_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)