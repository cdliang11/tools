# Copyright (c) 2024 Chengdong Liang(liangchengdongd@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from typing import Callable, Dict, Tuple, Union

from rms_norm_torch import RMSNorm
from rms_norm_triton import rms_norm_triton


def get_tol(dtype: torch.dtype) -> Dict:
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    elif dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-5, rtol=1.3e-6)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("seq_length", [128, 1024, 2048, 4096])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
def test_triton_rms_norm_fw(
    dtype: torch.dtype,
    batch_size: int,
    seq_length: int,
    hidden_size: int,
) -> None:
    device = torch.device("cuda:0")
    x = torch.rand((batch_size, seq_length, hidden_size),
                   dtype=dtype,
                   device=device)
    g = torch.rand(hidden_size, dtype=dtype, device=device)
    eps = 1e-5

    # torch
    model = RMSNorm(hidden_size, eps).cuda()
    model.eval()
    for p in model.parameters():
        p.data = g
    with torch.no_grad():
        output_te = model(x)

    # triton
    output_triton = rms_norm_triton(x, g, eps)

    torch.testing.assert_close(output_te, output_triton, **get_tol(dtype))
