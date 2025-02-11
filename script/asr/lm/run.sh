#!/usr/bin/env bash

stage=lm # lm/graph/decode/all
static=true
dir=exp/test_lm

. path.sh

mkdir -p $dir/graph $dir/model $dir/result

if [ $stage == "lm" ] || [ $stage == "all" ]; then
  lmplz -o 2 --text $dir/text --arpa $dir/lm.arpa --discount_fallback
fi


if [ $stage == "graph" ] || [ $stage == "all" ]; then
  e2e/make_graph.sh --e2e_unit $dir/unit.txt \
    --static $static \
    --lm $(realpath $dir/lm.arpa) $dir/graph
  # cp $resource/model/{final.zip,units.txt} $dir/model
  # cp $dir/graph/words.txt $dir/model
  # if $static; then
  #   graph="TLG.fst";
  #   cp $dir/graph/TLG.fst $dir/model
  # else graph=TL.fst
  #   graph="TL.fst,G.fst"
  #   cp $dir/graph/{TL.fst,G.fst} $dir/model
  # fi

# cat << EOF > $dir/model/online_asr.flag
# --chunk_size=16
# --model_path=final.zip
# --fst_path=${graph}
# --dict_path=words.txt
# --unit_path=units.txt
# EOF

fi


if [ $stage == "decode" ] || [ $stage == "all" ]; then
  online_decoder_main --chunk_size 16 \
    --wav_list $resource/testsets/test.list \
    --model_config_dir $dir/model \
    --num_threads=8 \
    --result $dir/result/hyp.txt \
    --logtostderr
  sed -i -e 's/.*\///' -e 's/.wav//g' $dir/result/hyp.txt
  python3 tools/compute-wer.py --char=1 --v=1 \
      $resource/testsets/test.txt \
      $dir/result/hyp.txt >$dir/result/wer
fi
