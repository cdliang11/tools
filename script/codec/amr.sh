#!/bin/bash

# Copyright (c) 2024 Chengdong Liang(liangchengdongd@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

wav_scp=""
tmp_wav_dir=""
amr_out_dir=""

for line in `cat $wav_scp`; do
    echo $line
    wav_name=`basename $line`
    wav_path=$line
    if [[ "${line##*.}" == "pcm" ]]; then
      echo "The file has a pcm extension."
      sox -r 16000 -b 16 -e signed-integer -c 1 -t raw  $line $tmp_wav_dir/${wav_name}.wav
      wav_path=$tmp_wav_dir/${wav_name}.wav
    fi
    amr_path=$amr_out_dir/${wav_name}.amr
    ffmpeg -i $wav_path -acodec amr_nb -ab 12.20k  -ar 8000 -ac 1 $amr_path
done
