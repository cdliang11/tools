#!/usr/bin/env bash

stage=decode # lm/graph/decode/all
static=true
dir=exp/aishell_fix
data=data/aishell
test_wav=/path/to/aishell1/wav/test/S0764/BAC009S0764W0121.wav


mkdir -p $dir/graph $dir/model $dir/result $data

# 这里举个特殊的例子，典型的情况有：
# 1. 用中文字去模拟英文发音
# 2. 用常用字去标注生僻字的发音
echo "case 情 况" > $dir/fix.dict


if [ $stage == "download" ] || [ $stage == "all" ]; then
  mkdir -p $data/model
  wget -P $data/model "https://wenet.org.cn/downloads?models=wenet&version=aishell_u2pp_conformer_libtorch.tar.gz"
  tar -zxvf $data/model/aishell_u2pp_conformer_libtorch.tar.gz -C $data/model
  mv $data/model/20210601_u2++_conformer_libtorch_aishell/* $data/model
  rm -r $data/model/20210601_u2++_conformer_libtorch_aishell/
fi

if [ $stage == "lm" ] || [ $stage == "all" ]; then
  rm -r $dir/train.txt
  for n in $(seq 1 10); do
    echo "甚至 出现 交易 几乎 停滞 的 case" >> $dir/train.txt
  done
  lmplz -o 2 --text $dir/train.txt --arpa $dir/lm.arpa --discount_fallback
fi


if [ $stage == "graph" ] || [ $stage == "all" ]; then
  e2e_wfst/make_graph.sh --e2e_unit $data/model/units.txt \
    --fix_dict $dir/fix.dict \
    --static $static \
    --lm $(realpath $dir/lm.arpa) $dir/graph
  cp $data/model/{final.zip,units.txt} $dir/model
  cp $dir/graph/words.txt $dir/model
  if $static; then
    graph="TLG.fst";
    cp $dir/graph/TLG.fst $dir/model
  else graph=TL.fst
    graph="TL.fst,G.fst"
    cp $dir/graph/{TL.fst,G.fst} $dir/model
  fi
fi


if [ $stage == "decode" ] || [ $stage == "all" ]; then
  export GLOG_logtostderr=1
  export GLOG_v=2
  export WENET_DIR=/jfs-hdfs/user/chengdong01.liang/workspace/github/wenet_release
  export BUILD_DIR=${WENET_DIR}/runtime/libtorch/build
  export OPENFST_BIN=${BUILD_DIR}/../fc_base/openfst-build/src
  export PATH=$PWD:${BUILD_DIR}/bin:${BUILD_DIR}/kaldi:${OPENFST_BIN}/bin:$PATH
  decoder_main --chunk_size -1 \
    --wav_path $test_wav \
    --model_path $dir/model/final.zip \
    --unit_path $dir/model/units.txt \
    --ctc_weight 0.5 \
    --rescoring_weight 1.0 \
    --thread_num 16 \
    --result $dir/result/hyp.txt \
    --fst_path $dir/model/TLG.fst \
    --beam 15.0 \
    --dict_path $dir/model/words.txt \
    --lattice_beam 7.5 \
    --max_active 7000 \
    --min_active 200 \
    --acoustic_scale 1.0 \
    --blank_skip_thresh 0.98 \
    --length_penalty 0.0

#   sed -i -e 's/.*\///' -e 's/.wav//g' $dir/result/hyp.txt
#   python3 tools/compute-wer.py --char=1 --v=1 \
#       $test_set/text \
#       $dir/result/hyp.txt >$dir/result/wer
fi
