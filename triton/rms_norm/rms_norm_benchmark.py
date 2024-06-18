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
import torch.nn.functional as F
import time
import triton

from rms_norm_torch import RMSNorm
from rms_norm_triton import rms_norm_triton


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[128 * i for i in range(2, 64)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[["blue", "-"], ["green", "-"]],
        ylabel="GB/s",
        plot_name="rms_norm_benchmark",
        args={"HIDDEN_SIZE": 128, "BATCH_SIZE": 1},
    ))
def benchmark_rms_norm(SEQ_LEN, HIDDEN_SIZE, BATCH_SIZE, provider):
    x = torch.rand((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE),
                   dtype=torch.float32,
                   device="cuda:0")
    g = torch.rand(HIDDEN_SIZE, dtype=torch.float32, device="cuda:0")
    eps = 1e-5
    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rms_norm_triton(x, g, eps), quantiles=quantiles)
    if provider == "torch":
        # torch
        model = RMSNorm(HIDDEN_SIZE, eps).to("cuda:0")
        model.eval()
        for p in model.parameters():
            p.data = g
        with torch.no_grad():
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: model(x), quantiles=quantiles)

    def gbps(ms):
        return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark_rms_norm.run(show_plots=False, print_data=True, save_path="./res_bs1")
