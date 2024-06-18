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

from rope_triton import apply_rotary_pos_emb as rope_triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[128 * i for i in range(2, 64)],
        line_arg="provider",
        line_vals=["triton", "transformer-engine"],
        line_names=["Triton", "Transformer-engine"],
        styles=[["blue", '-'], ["green", '-']],
        ylabel="GB/s",
        plot_name="rope_triton_benchmark",
        args={"HIDDEN_SIZE": 128, "BATCH_SIZE": 8, "HEAD_NUM": 64},
    ))
def benchmark_rope_triton(SEQ_LEN, HIDDEN_SIZE, BATCH_SIZE, HEAD_NUM, provider):
    x = torch.rand((SEQ_LEN, BATCH_SIZE, HEAD_NUM, HIDDEN_SIZE),
                   dtype=torch.float32,
                   device='cuda:0')
    rotary_pos_emb = RotaryPositionEmbedding(HIDDEN_SIZE)
    emb = rotary_pos_emb(SEQ_LEN)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rope_triton(x, emb, "sbhd"), quantiles=quantiles)
    if provider == 'transformer-engine':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: apply_rotary_pos_emb(x, emb, "sbhd"), quantiles=quantiles)

    def gbps(ms):
        return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark_rope_triton.run(show_plots=False,
                          print_data=True,
                          save_path="./res_bs8")
