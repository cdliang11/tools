#!/usr/bin/env bash

stage=graph # lm/graph/decode/all
static=true
dir=exp/aishell_slot
data=data/aishell


mkdir -p $dir/graph $dir/model $dir/result $data

# class lm 通过 slot 去实现，这里举个特殊的例子，典型的情况有：
# 1. {北京,上海, 西安, 苏州} 的天气
# 2. 拨打 $number, $number 可以通过 FST 实现
echo "北京" > $dir/city.slot
echo "上海" >> $dir/city.slot
echo "西安" >> $dir/city.slot
echo "苏州" >> $dir/city.slot
echo "city $dir/city.slot" > $dir/slot.config


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
    echo "\$city 的 天气 怎么样" >> $dir/train.txt
  done
  lmplz -o 2 --text $dir/train.txt --arpa $dir/lm.arpa --discount_fallback
fi


if [ $stage == "graph" ] || [ $stage == "all" ]; then
  e2e_wfst/make_graph.sh --e2e_unit $data/model/units.txt \
    --slot_config $dir/slot.config \
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
