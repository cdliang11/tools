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

import torch
import triton
import triton.language as tl

from transformer_engine.pytorch.attention import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)


@triton.jit
def _rope_kernel_fw(input_ptr,  # 输入数据的地址
                    in_seq_len_stride, in_batch_stride,  # 在batch / seq维度上对应的stride
                    output_ptr,  # 输出数据地址
                    cos_ptr, sin_ptr,  # cos, sin的地址
                    cos_stride, sin_stride,  # cos, sin的stride
                    seq_len, head_dim,
                    BLOCK_SIZE: tl.constexpr, BATCH_NUM: tl.constexpr):
    # 二维的grid
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)

    head_dim_offset = tl.arange(0, BLOCK_SIZE)  # [0, head_dim // 2]
    head_dim_mid = head_dim // 2

    mask = head_dim_offset < head_dim_mid

    # 计算cos, sin的offset
    cos_offset = (pid_seq % seq_len) * cos_stride + head_dim_offset
    sin_offset = (pid_seq % seq_len) * sin_stride + head_dim_offset

    cos = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0)

    for batch_idx in tl.static_range(0, BATCH_NUM):
        x1_offset = pid_seq * in_seq_len_stride + batch_idx * in_batch_stride \
                    + pid_head * head_dim + head_dim_offset
        x2_offset = pid_seq * in_seq_len_stride + batch_idx * in_batch_stride \
                    + pid_head * head_dim + head_dim_mid + head_dim_offset

        x1 = tl.load(input_ptr + x1_offset, mask=mask, other=0.0)
        x2 = tl.load(input_ptr + x2_offset, mask=mask, other=0.0)

        # rope的核心计算
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        # 保存数据
        tl.store(output_ptr + x1_offset, y1, mask=mask)
        tl.store(output_ptr + x2_offset, y2, mask=mask)

    return


def fused_rope_triton(inp, freqs, tensor_format):
    if tensor_format == 'bshd':
        inp = inp.transpose(0, 1)
    elif tensor_format != 'sbhd':
        raise ValueError(f"Unsupported tensor format: {tensor_format}.")

    seq_len, batch_num, head_num, head_dim = inp.shape
    output = torch.empty_like(inp)

    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)

    grid = (seq_len, head_num)

    freqs = freqs[:seq_len]

    # 提前算好cos sin
    cos = torch.cos(freqs).to(inp.dtype)
    sin = torch.sin(freqs).to(inp.dtype)

    _rope_kernel_fw[grid](inp,
                          inp.stride(0),
                          inp.stride(1),
                          output,
                          cos,
                          sin,
                          cos.stride(0),
                          sin.stride(0),
                          seq_len,
                          head_dim,
                          BLOCK_SIZE,
                          batch_num)

    if tensor_format == 'bshd':
        output = output.transpose(0, 1)
    return output


def apply_rotary_pos_emb(
        inp: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
):
    return fused_rope_triton(inp, freqs, tensor_format)

