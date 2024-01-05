# Copyright (c) 2022 Chengdong Liang(liangchengdongd@qq.com)
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

# 对数据集进行vad处理 同时保持原有的文件结构组织

import os
import argparse
import glob2
import pypeln as pl
from tqdm import tqdm
from tools.vad_process import filter


def vad_main(wav_list):
    old_wav, new_wav = wav_list[0], wav_list[1]
    filter(old_wav, new_wav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vad')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--re', required=True)
    parser.add_argument('--nj', type=int, default=16)
    args = parser.parse_args()

    # glob
    wavs = glob2.glob(os.path.join(args.dataset, '**/*.wav'))
    re = args.re
    wav_list = []
    for wav in wavs:
        old_wav = wav
        new_wav = wav.replace(re, re + '_vad')
        if not os.path.exists(os.path.dirname(new_wav)):
            os.makedirs(os.path.dirname(new_wav))
        wav_list.append([old_wav, new_wav])
    lines_num = len(wav_list)
    t_bar = tqdm(ncols=100, total=lines_num)
    for _ in pl.process.map(vad_main,
                            wav_list,
                            workers=args.nj,
                            maxsize=args.nj + 1):
        t_bar.update()

    t_bar.close()
