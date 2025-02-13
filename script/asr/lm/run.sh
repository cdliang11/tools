#!/usr/bin/env bash

stage=graph # lm/graph/decode/all
static=true
dir=exp/test_lm
data=data/toy

mkdir -p $dir/graph

if [ $stage == "lm" ] || [ $stage == "all" ]; then
  lmplz -o 2 --text $data/text --arpa $dir/lm.arpa --discount_fallback
fi


if [ $stage == "graph" ] || [ $stage == "all" ]; then
  e2e_wfst/make_graph.sh --e2e_unit $data/unit.txt \
    --static $static \
    --lm $(realpath $dir/lm.arpa) $dir/graph
fi
