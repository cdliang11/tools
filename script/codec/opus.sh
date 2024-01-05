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
pcm_out_dir=""
opus_out_dir=""

for line in `cat $wav_scp`; do
    echo $line
    if [[ "${line##*.}" == "pcm" ]]; then
      echo "The file has a pcm extension."
      wav_name=`basename $line`
      cp -r $line $pcm_out_dir
      opus_path=$opus_out_dir/${wav_name}.opus
      opus_demo -e restricted-lowdelay  16000  1  -framesize=10 $line  ${opus_path}
    else
      wav_name=`basename $line`
      echo $wav_name
      pcm_path=$pcm_out_dir/${wav_name}.pcm
      ffmpeg -y -i $line -acodec pcm_s16le -f s16le -ac 1 -ar 16000 $pcm_path
      opus_path=$opus_out_dir/${wav_name}.opus
      opus_demo -e restricted-lowdelay  16000  1  -framesize=10  ${pcm_path} ${opus_path}
    fi
done
