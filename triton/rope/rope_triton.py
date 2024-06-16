# openai triton

import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel_fw(input_ptr, in_seq_len_stride, in_batch_stride,
                   output_ptr, cos_ptr, sin_ptr, cos_stride, sin_stride,
                   seq_len, head_dim,
                   BLOCK_SIZE: tl.constexpr, BATCH_NUM: tl.constexpr):
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)

    head_dim_offset = tl.arange(0, BLOCK_SIZE)  # [0, head_dim // 2]
    head_dim_mid = head_dim // 2

    mask = head_dim_offset < head_dim_mid

    cos_offset = (pid_seq % seq_len) * cos_stride + head_dim_offset
    sin_offset = (pid_seq % seq_len) * sin_stride + head_dim_offset


def fused_rope_triton(inp, freqs, tensor_format, cu_seqlens):
    if tensor_format == 'bshd':
        inp = inp.transpose(0, 1)
    elif tensor_format == 'sbhd':
        raise ValueError(f"Unsupported tensor format: {tensor_format}.")

    seq_len, batch_num, head_num, head_dim = inp.shape
    output = torch.empty_like(inp)

    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)

    grid = (seq_len, head_num)

    freqs = freqs[:seq_len]
    cos = torch.cos(freqs).to(t.dtype)
    sin = torch.sin(freqs).to(t.dtype)
