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


@triton.jit
def rms_norm_kernel_fw(X, Y, W, Rstd, stride_B, stride_S, S, D, eps, BLOCK_SIZE: tl.constexpr):
    """Implements the forward pass of the RMSNorm operation.
    Parameters:
        X: [B, S, D] input tensor where each column represents a feature.
        Y: output tensor for normalized features.
        W: weights for scaling the normalized data
        Rstd: Tensor to store reciprocal of the computed standard deviations.
        stride_B: batch stride of X.
        stride_S: sequence stride of X.
        S: X.size(1)
        D: X.size(2)
        eps: small epsilon value for numerical stability in division.
        BLOCK_SIZE: Number of threads in a block.
    """
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)

    base_idx = pid_batch * stride_B + pid_seq * stride_S
    Y += base_idx
    X += base_idx

    _rms = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < D, other=0.0).to(tl.float32)
        _rms += a * a

    rms = tl.sqrt(tl.sum(_rms) / D + eps)

    # 存储标准差
    tl.store(Rstd + pid_batch * S + pid_seq, rms)

    for off in range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        w = tl.load(W + cols, mask=mask, other=0.0)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x / rms
        y = x_hat * w
        tl.store(Y + cols, y, mask=mask)


def rms_norm_triton(x, g, eps=1e-5):
    B, S, D = x.shape  # M, L, N
    y = torch.empty_like(x, dtype=torch.float32, device=x.device)
    rstd = torch.empty(B * S, dtype=torch.float32, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(D)

    grid = (B, S)
    rms_norm_kernel_fw[grid](x, y, g, rstd, x.stride(0), x.stride(1), S, D, eps, BLOCK_SIZE)

    return y
