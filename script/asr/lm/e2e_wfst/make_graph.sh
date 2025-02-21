#!/usr/bin/env bash

stage=0
stop_stage=100
lm=
e2e_unit=
bpe_model=
slot_config=
fix_dict=
static=true

. tools/parse_options.sh

if [ $# -ne 1 ]; then
  echo "make_graph.sh <dir>"
  exit 1
fi

dir=$1

mkdir -p $dir

[ ! -f $lm ] && echo "lm file $lm not exist, please check!!!" && exit 1;
[ ! -f $e2e_unit ] && echo "unit file not exist, please check!!!" && exit 1;

# Step 1. Extract vocab for graph building, including words in LM and class list
# Extracted vocab:
# a. lm.vocab: vocab from LM
# b. class.vocab: vocab from class list
# c. lexicon.vocab: union(lm.vocab, class.vocab)
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  python3 e2e_wfst/dict/extract_vocab.py \
    ${slot_config:+--slot_config $slot_config} \
    $lm $dir
fi


# Step 2. Prepare lexicon and input/output symbols
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  python3 e2e_wfst/dict/prepare_dict.py \
    ${fix_dict:+--fix_dict $fix_dict} \
    ${bpe_model:+--bpe_model $bpe_model} \
    $e2e_unit $dir/lexicon.vocab $dir/lexicon.txt
  ndisambig=$(e2e_wfst/fst/add_lex_disambig.pl $dir/lexicon.txt \
      $dir/lexicon_disambig.txt)
  python3 e2e_wfst/fst/prepare_symbols.py \
    ${slot_config:+--slot_config $slot_config} $e2e_unit \
    $dir/lexicon_disambig.txt $ndisambig $dir/tokens.txt $dir/words.txt
fi

# Step 3. Build FST T/L/G, respectively
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # 3.1 T.fst
  python3 e2e_wfst/fst/ctc_token_fst_compact.py $dir/tokens.txt | \
    fstcompile --isymbols=$dir/tokens.txt --osymbols=$dir/tokens.txt | \
    fstarcsort --sort_type=olabel > $dir/T.fst || exit 1;

  # 3.2 L.fst
  python3 e2e_wfst/fst/make_lexicon_fst.py \
    $dir/lexicon_disambig.txt $dir/lexicon.fst.txt
  grep "\#" $dir/tokens.txt | awk '{print $2}' > $dir/t.disambig.list
  grep "\#" $dir/words.txt | awk '{print $2}' > $dir/w.disambig.list
  fstcompile --isymbols=$dir/tokens.txt --osymbols=$dir/words.txt \
    --keep_isymbols=false --keep_osymbols=false $dir/lexicon.fst.txt | \
    fstaddselfloops $dir/t.disambig.list $dir/w.disambig.list | \
    fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;

  # 3.3 G.fst
  # 3.3.1 prepare G.fst
  arpa2fst --read-symbol-table=$dir/words.txt --disambig-symbol=#0 $lm - |
      fstarcsort --sort_type=ilabel > $dir/G.fst

  # 3.3.2 Optional write slot FST and do replace on G.
  if [ ! -z $slot_config ]; then
    mv $dir/G.fst $dir/G.lm.fst
    python3 e2e_wfst/fst/write_slot_and_replace.py $slot_config \
      $dir/words.txt $dir/G.lm.fst $dir/G.fst
  fi
fi


# Step 4. Build TLG.fst or TL/G.fst
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  if $static; then
    echo "Compile static TLG graph"
    fsttablecompose $dir/L.fst $dir/G.fst | \
      fstdeterminizestar --use-log=true | \
      fstminimizeencoded | \
      fstarcsort --sort_type=ilabel > $dir/LG.fst
    fsttablecompose $dir/T.fst $dir/LG.fst > $dir/TLG.fst
    #fstconvert --fst_align --fst_type=const > $dir/TLG.fst
  else
    echo "Compile dynamic TL/G graph"
    # Create olabel lookahead TL.fst
    mv $dir/L.fst $dir/Lr.fst   # r means raw
    fstdeterminizestar $dir/Lr.fst | fstminimizeencoded | \
      fstarcsort --sort_type=ilabel > $dir/L.fst
    fsttablecompose $dir/T.fst $dir/L.fst | \
      fstarcsort --sort_type=olabel | \
      fstconvert --fst_type=olabel_lookahead \
        --save_relabel_opairs=$dir/relabel.txt > $dir/TL.fst

    mv $dir/G.fst $dir/Gr.fst   # r means raw
    # relabel ilabel of G.fst
    fstrelabel --relabel_ipairs=$dir/relabel.txt $dir/Gr.fst | \
      fstarcsort --sort_type=ilabel | \
      fstconvert --fst_align --fst_type=const > $dir/G.fst
  fi
fi
